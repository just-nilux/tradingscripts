import json
import decimal
import pandas as pd

from time import time
from time import sleep
from dydx3 import Client
from typing import Tuple, List, Dict, Any
from logger_setup import setup_logger
from dydx3.constants import TIME_IN_FORCE_IOC, ORDER_TYPE_MARKET



class DydxClient:
    def __init__(self):

        with open('config.json') as config_file:
            self.config = json.load(config_file)

        with open('asset_config.json') as f:
            self.asset_resolution = json.load(f)


        self.api_key = self.config['dydx_api_key']
        self.secret_key = self.config['dydx_api_secret']
        self.passphrase = self.config['dydx_passphrase']
        self.stark_private_key = self.config['dydx_stark_private_key']
        self.default_ethereum_address = self.config['dydx_default_ethereum_address']

        self.logger = setup_logger(__name__)



        self.client, self.position_id = self.initialize_dydx_client(
            self.api_key,
            self.secret_key,
            self.passphrase,
            self.stark_private_key,
            self.default_ethereum_address
        )

    
    def initialize_dydx_client(self, api_key: str, secret_key: str, passphrase: str, stark_priv_key:str, eth_address: str) -> Tuple[Client, str]:
        """
        Initializes a dYdX API client and gets the position ID for the specified Ethereum address.
        
        Parameters:
        api_key (str): Your dYdX API key.
        secret_key (str): Your dYdX API secret key.
        passphrase (str): Your dYdX API passphrase.
        stark_priv_key (str): Your dYdX API Stark private key.
        eth_address (str): The Ethereum address associated with your dYdX account.
        
        Returns:
        A tuple containing the dYdX API client and the position ID.

        Example: -> client, position_id = initialize_dydx_client(API_KEY, SECRET_KEY, PASSPHRASE, STARK_PRIV_KEY, ETH_ADDRESS)
        """
        try:
            client = Client(
                host="https://api.dydx.exchange",
                api_key_credentials={
                    'key': api_key,
                    'secret': secret_key,
                    'passphrase': passphrase
                },
                stark_private_key=stark_priv_key,
                default_ethereum_address=eth_address
            )
            
            account_response = client.private.get_account()
            position_id = account_response.data['account']['positionId']
            
            return client, position_id

        except Exception as e:
            self.logger.error(f"An error occurred while initialize dydx client: {e}")
            return None
        



    def fetch_all_symbols(self) -> List[str]:
        """
        Fetches all available market symbols from the dYdX API.

        Returns:
        List of strings representing all available market symbols.
        """
        try:
            market_data = self.client.public.get_markets()
            symbols = list(market_data.data['markets'].keys())

            return symbols
        
        except Exception as e:
            self.logger.error(f"An error occurred while fetching all symbols: {e}")
            return None



    def fetch_free_equity(self) -> float:
        """
        Fetches the amount of free collateral available in the account.
        
        Returns:
            A float value of the amount of free collateral in the account.
        """
        try:
            data = self.client.private.get_account().data
            free_collateral = float(data['account']['freeCollateral'])

            return free_collateral

        except Exception as e:
            self.logger.error(f"An error occurred while fetching free equity: {e}")
            return None
        


    def format_positions_data(self, positions_data) -> str:
        """
        Format positions data into a string.

        Args:
            positions_data (dict): The positions data to format.

        Returns:
            str: Formatted positions data.
        """
        if not positions_data or not positions_data.get('positions'):
            return "No open positions"

        result_str = ""
        for position in positions_data.get('positions', []):
            result_str += f"\n\nMarket: {position['market']}\n"
            result_str += f"Side: {position['side']}\n"
            result_str += f"Size: {position['size']}\n"
            result_str += f"Entry Price: {position['entryPrice']}\n"
            result_str += f"Unrealized PnL: {position['unrealizedPnl']}\n"
            result_str += f"Created At: {position['createdAt']}"
        
        return result_str



    def fetch_all_open_position(self, symbol=None) -> str:
        """
        Fetch all open positions for the authenticated user, filtered by a specific symbol if provided.

        Args:
            symbol (str, optional): The trading pair symbol to filter the positions (e.g., 'BTC-USD', 'ETH-USD'). 
                                    If not provided, returns all open positions across all symbols.

        Returns:
            str: A string containing information about open positions for the specified symbol or all symbols if not provided.

        """
        try:
            if symbol is None:
                positions_data = self.client.private.get_positions(status='OPEN').data
            elif isinstance(symbol, str):
                positions_data = self.client.private.get_positions(market=symbol, status='OPEN').data

            return self.format_positions_data(positions_data)

        except Exception as e:
            return "An error occurred while fetching open positions"



    def get_open_orders(self):
        response = self.client.private.get_orders(status='OPEN').data
        return response



    def get_klines(self, symbol: str, interval: str):
        """
        Fetch and return candlestick data for a given symbol and interval.

        Parameters:
        symbol (str): The trading pair symbol to fetch candlestick data for.
        interval (str): The time interval for each candlestick.

        Returns:
        pd.DataFrame: A pandas DataFrame containing the candlestick data with columns: 'Open', 'High', 'Low', 'Close', and 'Volume'.

        This function continuously attempts to fetch candlestick data from the public dydx API until successful.
        If an exception occurs during fetching, it waits for 0.5 seconds before retrying.
        """
        while True:
            try:
                candles = self.client.public.get_candles(market=symbol, resolution=interval)

                df = pd.DataFrame(candles.data['candles'])[['open', 'high', 'low', 'close', 'usdVolume']]
                df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'usdVolume': 'Volume'}).astype(float)

                return df
            
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
                self.logger.error("Retrying in 0.5 seconds...")
                sleep(0.25)



    def order_size(self, symbol: str, position_size: float):
        """
        Calculate the order size based on the available equity, leverage, and market data.

        Parameters:
        symbol (str): The symbol of the asset for which the order size is to be calculated.
        position_size (float): The proportion of free equity to use for the order.

        Returns:
        float: The calculated order size.

        This function first retrieves the leverage for the specified symbol and calculates the 
        amount of free equity that should be used for the order. It then fetches the current 
        market data for the symbol, including the oracle price and the minimum order size.

        The function then calculates the order size based on the free equity, the leverage, and 
        the oracle price, ensuring that the order size is a multiple of the minimum order size. 
        It returns this calculated order size.
        """


        leverage = self.config['leverage'][symbol]
        free_equity = self.fetch_free_equity() * position_size
        market_data = self.client.public.get_markets(market=symbol).data['markets'][symbol]
        price = float(market_data['oraclePrice'])
        min_order_size = float(market_data['minOrderSize'])

        return float(int((int(1 / min_order_size)) * (free_equity / price * leverage)) / int(1 / min_order_size))
    


    def calculate_tp_sl(self, price: float, atr: float, side: str, trigger_candle: pd.Series, tick_size: float, risk_to_reward_ratio=None):
        """
        Calculate the take profit and stop loss levels based on the current price, ATR, side of trade, 
        trigger candle data, tick size, and the risk to reward ratio.

        Parameters:
        price (float): The current price of the asset.
        atr (float): The current Average True Range (ATR) value.
        side (str): The side of the trade, either "BUY" or "SELL".
        trigger_candle (pd.Series): The series containing the high and low values of the trigger candle.
        tick_size (float): The minimum price movement of the asset.
        risk_to_reward_ratio (float, optional): The desired risk to reward ratio. If not provided, the default value from the configuration is used.

        Returns:
        list: A list containing the take profit level and the stop loss level.

        This function calculates the take profit (tp) and stop loss (sl) levels based on the current price,
        the Average True Range (ATR), and the risk to reward ratio. The trigger candle's high and low values 
        are used in the calculation, depending on whether the side of the trade is "BUY" or "SELL". 

        The calculated tp and sl are adjusted to be multiples of the tick size. 
        The function then returns the tp and sl as a list.
        """
        if risk_to_reward_ratio is None:
            risk_to_reward_ratio = self.config['strategies'][0]['risk_to_reward_ratio']

        atr_value = atr * 1
        
        if side == 'BUY':
            stop_loss = trigger_candle.low - atr_value
            take_profit = price + (price - stop_loss) * risk_to_reward_ratio
        elif side == "SELL":
            stop_loss = trigger_candle.high + atr_value
            take_profit = price - (stop_loss - price) * risk_to_reward_ratio

        take_profit = round(take_profit / tick_size) * tick_size
        stop_loss = round(stop_loss / tick_size) * tick_size
        
        return [take_profit, stop_loss]
    


    def place_market_order(self, symbol: str, size: float, side: str, atr: float, trigger_candle: pd.Series, percent_slippage=0.003) -> str:
        """
        Place a market order for a specified symbol, size, and side, considering the specified percent slippage.

        Args:
            symbol (str): The trading pair symbol (e.g., 'BTC-USD', 'ETH-USD').
            size (float): The size of the asset to buy or sell.
            side (str): The order side ('BUY' or 'SELL').
            percent_slippage (float, optional): The maximum allowed slippage percentage. Defaults to 0.003 (0.3%).

        Returns:
            str: The ID of the created order.

        """
        try:
            market_data = self.client.public.get_markets(market=symbol).data
            price = float(market_data['markets'][symbol]['oraclePrice'])
        
            if side == 'BUY':
                price = price * (1 + percent_slippage) # max allowed price incl slippage
                tpsl_side = 'SELL'
            elif side == 'SELL':
                price = price * (1 - percent_slippage) # min allowed price incl slippage
                tpsl_side = 'BUY'
            else:
                raise ValueError("Invalid side value. Must be 'BUY' or 'SELL'.")

            tick_size = float(market_data['markets'][symbol]['tickSize'])
            price = round(price / tick_size) * tick_size
            decimals = abs((decimal.Decimal(market_data['markets'][symbol]['indexPrice']).as_tuple().exponent))
            expiration_epoch_seconds = int((pd.Timestamp.utcnow() + pd.Timedelta(weeks=1)).timestamp())

            order_params = {
                'position_id': self.position_id, 
                'market' : symbol,
                'side' : side,
                'order_type' : ORDER_TYPE_MARKET,
                'post_only': False,
                'size' : str(size),
                'price' : str(round(price, decimals)),
                'limit_fee' : '0.0015',
                'expiration_epoch_seconds' : expiration_epoch_seconds,
                'time_in_force' : TIME_IN_FORCE_IOC,
                }

            order_response = self.client.private.create_order(**order_params)
            order_id = order_response.data['order']['id']


            # Stop-Loss & Take-Profit order:

            TPSL_ORDER_TYPE = ['TAKE_PROFIT', 'STOP_LIMIT']
            tpsl = self.calculate_tp_sl(price, atr, side, trigger_candle, tick_size)

            for i in range(2):
                formatted_price = str(round(tpsl[i], decimals)) 
                order_params['side'] = tpsl_side
                order_params['order_type'] = TPSL_ORDER_TYPE[i]
                order_params['reduce_only'] = True
                # Set a wide limit price for stop-limit orders to mimic a stop-market order
                if TPSL_ORDER_TYPE[i] == 'STOP_LIMIT':
                    wide_limit_price = round((tpsl[i] * 0.95)/ tick_size)*tick_size if tpsl_side == 'SELL' else round((tpsl[i] * 1.05)/tick_size)*tick_size
                    order_params['price'] = str(round(wide_limit_price, decimals))
                else:
                    order_params['price'] = formatted_price
                order_params['trigger_price'] = formatted_price
                order_params['time_in_force'] = TIME_IN_FORCE_IOC

                order_response = self.client.private.create_order(**order_params)
                order_id = order_response.data['order']['id']


            #return order_id

        except Exception as e:
            self.logger.error(f"An error occurred while placing the market order: {e}")
            return None
            