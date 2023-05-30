import json
import decimal
import pandas as pd
from datetime import datetime

from time import time
from time import sleep
from dydx3 import Client
from typing import Tuple, List, Optional
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

        self.init_no_of_trades = len(self.client.private.get_positions(status='Closed').data['positions'])

    
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


    def send_tg_msg_when_trade_closed(self):
        """
        Checks if a trade has been closed since the last check. If a trade has been closed, 
        it formats the relevant information about the trade into a message and returns it.
        The message can then be sent to Telegram or used for other purposes.
    
        Returns:
            str: A formatted message containing information about the closed trade, 
            or None if no trade has been closed since the last check.
        """


        current_no_of_historical_trade = len(self.client.private.get_positions(status='Closed').data['positions'])

        if current_no_of_historical_trade == self.init_no_of_trades:
            return
        
        elif current_no_of_historical_trade > self.init_no_of_trades:
            self.init_no_of_trades +=1
            position = self.client.private.get_positions(status='Closed').data['positions'][0]

            msg = (
                f"*** TRADE CLOSED ***\n"
                f"Opened: {datetime.fromisoformat(position['createdAt'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Closed: {datetime.fromisoformat(position['closedAt'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Market: {position['market']}\n"
                f"Status: {position['status']}\n"
                f"Side: {position['side']}\n"
                f"Size: {position['maxSize']}\n"
                f"Entry Price: {position['entryPrice']}\n"
                f"Exit Price: {position['exitPrice']}\n"
                #f"Unrealized PnL: {position['unrealizedPnl']}\n"
                f"Realized PnL: {position['realizedPnl']}\n"
                #f"Net Funding: {position['netFunding']}\n"
            )

            return msg
        

    def cancel_order_by_symbol(self, symbol):
        self.logger.info(f'remaining open orders for {symbol} have been cancelled')
        return self.client.private.cancel_all_orders(symbol).data





    def purge_no_pos_orders(self):
        """
        Cancels all orders for trading pairs which don't have an open position.

        The method fetches all open positions and all orders, determines the trading pairs 
        (symbols) for which there are orders but no open positions, and cancels all such orders. 

        Returns:
            dict: Response from the cancel_all_orders API call for the last symbol in the purge_symbols list.
            Returns None if an error occurred during the process.

        Raises:
            Exception: If an error occurs while fetching positions or orders or cancelling orders, an exception is logged and None is returned.
        """

        try:

            positions = self.client.private.get_positions(status='Open').data['positions']
            symbols_pos = [position['market'] for position in positions]

            orders = self.client.private.get_orders().data['orders']
            symbols_orders = [order['market'] for order in orders]
            
            purge_symbols = list(set(symbols_orders) - set(symbols_pos))
            for sym in purge_symbols:
                res = self.client.private.cancel_all_orders(sym).data
                self.logger.info(f'remaining open orders for {sym} have been cancelled - not in position anymore')
        
        except Exception as e:
            self.logger.error(f"An error occurred while fetching all symbols: {e}")
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
        


    def fetch_all_open_position(self, symbol=None, first_position=False) -> str:
        """
        Fetch all open positions for the authenticated user, filtered by a specific symbol if provided.

        Args:
            symbol (str, optional): The trading pair symbol to filter the positions (e.g., 'BTC-USD', 'ETH-USD'). 
                                    If not provided, returns all open positions across all symbols.

        Returns:
            str: A string containing information about open positions for the specified symbol or all symbols if not provided.

        """
        try:
            # Fetch positions and orders data
            orders_data = self.client.private.get_orders().data['orders']

            if first_position and not symbol:
                positions_data = self.client.private.get_positions(status='OPEN').data

                # Get the first open position
                first_position = positions_data['positions'][0]

                # Build a dictionary of oracle prices, stop limit and take profit prices for the market of the first open position
                market_prices = {
                    first_position['market']: {
                        'oracle': self.client.public.get_markets(first_position['market']).data['markets'][first_position['market']]['oraclePrice'],
                        'STOP_LIMIT': next((order['triggerPrice'] for order in orders_data if order['market'] == first_position['market'] and order['type'] == 'STOP_LIMIT'), None),
                        'TAKE_PROFIT': next((order['price'] for order in orders_data if order['market'] == first_position['market'] and order['type'] == 'TAKE_PROFIT'), None)
                    }
                }
                return self.format_positions_data(positions_data, market_prices, first_position = True)
            
            else:
                positions_data = self.client.private.get_positions(market=symbol, status='OPEN').data if symbol else self.client.private.get_positions(status='OPEN').data
            
                # Build a dictionary of oracle prices, stop limit and take profit prices for each market
                market_prices = {
                    position['market']: {
                        'oracle': self.client.public.get_markets(position['market']).data['markets'][position['market']]['oraclePrice'],
                        'STOP_LIMIT': next((order['triggerPrice'] for order in orders_data if order['market'] == position['market'] and order['type'] == 'STOP_LIMIT'), None),
                        'TAKE_PROFIT': next((order['price'] for order in orders_data if order['market'] == position['market'] and order['type'] == 'TAKE_PROFIT'), None)
                    }
                    for position in positions_data['positions']
                }

                return self.format_positions_data(positions_data, market_prices, first_position)

        except Exception as e:
            return "An error occurred while fetching open positions"


    def format_positions_data(self, positions_data, market_prices, first_position) -> str:
        """
        Format positions data into a string.

        Args:
            positions_data (dict): The positions data to format.
            market_prices (dict): A dictionary of oracle prices, stop limit and take profit prices for each market.

        Returns:
            str: Formatted positions data.
        """
        if not positions_data or not positions_data.get('positions'):
            return "No open positions"

        results = []
        for position in positions_data.get('positions', []):
            current_prices = market_prices.get(position['market'], {})
            entry_price = float(position['entryPrice'])
            tp_price = float(current_prices.get('TAKE_PROFIT', 0))
            sl_price = float(current_prices.get('STOP_LIMIT', 0))

            tp_percent_change = ((tp_price - entry_price) / entry_price) * 100 if entry_price else None
            sl_percent_change = ((sl_price - entry_price) / entry_price) * 100 if entry_price else None

            # Set the optional string based on the 'first_position' flag
            first_position_string = "\n*** OPENED POSITION ***\n" if first_position else ""

            results.append(
                f"{first_position_string}"
                f"\nOpened At: {datetime.fromisoformat(position['createdAt'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Market: {position['market']}\n"
                f"Side: {position['side']}\n"
                f"Size: {position['size']}\n"
                f"Entry Price: {position['entryPrice']}\n"
                f"TP: {current_prices.get('TAKE_PROFIT')} ({tp_percent_change:.2f}%)\n"
                f"SL: {current_prices.get('STOP_LIMIT')} ({sl_percent_change:.2f}%)\n"
                f"Current Price: {current_prices.get('oracle')}\n"
                f"Unrealized PnL: {position['unrealizedPnl']}\n"
            )
        return "".join(results)



    def get_open_orders(self):
        response = self.client.private.get_orders().data
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


    def order_size(self, symbol: str, position_size: float) -> Optional[float]:
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
        try:
            leverage = self.config['leverage']
            if leverage is None:
                raise ValueError(f"Leverage not found for symbol: {symbol}")

            if not 0 < position_size <= 1:
                raise ValueError("position_size should be a value between 0 and 1")

            free_equity = self.fetch_free_equity() * position_size
            market_data = self.client.public.get_markets(market=symbol).data.get('markets', {}).get(symbol)
            if market_data is None:
                raise ValueError(f"Market data not found for symbol: {symbol}")

            price = float(market_data.get('oraclePrice', 0))
            if price <= 0:
                raise ValueError(f"Invalid oracle price: {price}")

            min_order_size = float(market_data.get('minOrderSize', 0))
            if min_order_size <= 0:
                raise ValueError(f"Invalid minimum order size: {min_order_size}")

            calculated_order_size = float(int((int(1 / min_order_size)) * (free_equity / price * leverage)) / int(1 / min_order_size))
            self.logger.info(f"Calculated order size for {symbol} is {calculated_order_size}")
            return calculated_order_size

        except Exception as e:
            self.logger.error(f"Error calculating order size for {symbol}: {e}")
            return None




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

        atr_value = atr * 2
        
        if side == 'BUY':
            stop_loss = trigger_candle.low - atr_value
            take_profit = price + (price - stop_loss) * risk_to_reward_ratio
        elif side == "SELL":
            stop_loss = trigger_candle.high + atr_value
            take_profit = price - (stop_loss - price) * risk_to_reward_ratio

        take_profit = round(take_profit / tick_size) * tick_size
        stop_loss = round(stop_loss / tick_size) * tick_size
        
        return [take_profit, stop_loss]
    
    

    def place_market_order(self, symbol: str, size: float, side: str, atr: float, trigger_candle: pd.Series, percent_slippage=0.003) -> Optional[str]:
        """
        Place a market order for a specified symbol, size, and side, considering the specified percent slippage.

        Args:
            symbol (str): The trading pair symbol (e.g., 'BTC-USD', 'ETH-USD').
            size (float): The size of the asset to buy or sell.
            side (str): The order side ('BUY' or 'SELL').
            percent_slippage (float, optional): The maximum allowed slippage percentage. Defaults to 0.003 (0.3%).

        Returns:
            Optional[str]: The ID of the created order or None if an error occurred.

        """
        try:
            market_data = self.client.public.get_markets(market=symbol).data.get('markets', {}).get(symbol)
            if market_data is None:
                raise ValueError(f"Market data not found for symbol: {symbol}")

            price = float(market_data.get('oraclePrice', 0))
            if price <= 0:
                raise ValueError(f"Invalid oracle price: {price}")

            tick_size = float(market_data.get('tickSize', 0))
            if tick_size <= 0:
                raise ValueError(f"Invalid tick size: {tick_size}")

            if side not in ['BUY', 'SELL']:
                raise ValueError("Invalid side value. Must be 'BUY' or 'SELL'.")

            # Adjust price based on slippage and side
            if side == 'BUY':
                price = price * (1 + percent_slippage) # max allowed price incl slippage
                tpsl_side = 'SELL'
            elif side == 'SELL':
                price = price * (1 - percent_slippage) # min allowed price incl slippage
                tpsl_side = 'BUY'

            # Normalize price to tick size
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

            if not order_id:
                self.logger.error(f'Order for {symbol} did not go through')
                return

            # Stop-Loss & Take-Profit order:
            order_ids = list()
            order_ids.append('ENTRY')

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
                order_ids.append(TPSL_ORDER_TYPE[i])


            return order_ids

     
        except Exception as e:
            self.logger.error(f"Error placing market order for {symbol}: {e}")
            return None



    def send_tg_msg_when_pos_opened(self, symbol= None):
        """
        Formats the relevant information about a newly opened trade into a message 
        and returns it. The message can then be sent to Telegram or used for other purposes.

        Returns:
            str: A formatted message containing information about the opened trade.
        """

        #position = self.client.private.get_positions(status='Open').data['positions'][0]

        #msg = (
        #    f"*** TRADE OPENED ***\n"
        #    f"Opened At: {datetime.fromisoformat(position['createdAt'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')}\n"
        #    f"Market: {position['market']}\n"
        #    f"Status: {position['status']}\n"
        #    f"Side: {position['side']}\n"
        #    f"Size: {position['maxSize']}\n"
        #    f"Entry Price: {position['entryPrice']}\n"
        #    f"Unrealized PnL: {position['unrealizedPnl']}\n"
        #)

        return self.fetch_all_open_position(symbol=symbol)



            