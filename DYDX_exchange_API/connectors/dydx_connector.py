import json
import decimal
import pandas as pd

from time import time
from dydx3 import Client

with open('config.json') as config_file:
    config = json.load(config_file)

client = Client(host = 'https://api.dydx.exchange', api_key_credentials = {'key': config['dydx_api_key'], 'secret': config['dydx_api_secret'], 'passphrase': config['dydx_passphrase']}, stark_private_key = config['dydx_stark_private_key'], default_ethereum_address = config['dydx_default_ethereum_address'])

TPSL_ORDER_TYPE = ['TAKE_PROFIT', 'STOP_LIMIT']

def get_client():
    return client

def get_all_symbols():
    return list(client.public.get_markets().data['markets'])

def get_wallet_balance():
    return float(client.private.get_account().data['account']['freeCollateral'])

def get_klines(symbol: str, interval: str):
    candles = client.public.get_candles(market = symbol, resolution = interval)

    df = pd.DataFrame(candles.data['candles'])[['open', 'high', 'low', 'close', 'usdVolume']]
    df = df.rename(columns = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'usdVolume': 'Volume'})

    return df

def order_size(symbol: str, symbol_num: int, strat: int):
    free_equity = get_wallet_balance() * config['position_size']
    market_data = client.public.get_markets(market = symbol).data['markets'][symbol]
    price = float(market_data['oraclePrice'])
    min_order_size = float(market_data['minOrderSize'])

    return float(int((int(1 / min_order_size)) * (free_equity / price * config['leverage'][symbol_num][strat])) / int(1 / min_order_size))

def place_order(symbol: str, size: float, side: str, order_type: str, df: pd.DataFrame):
    side = side.upper()
    order_type = order_type.upper()

    position_id = client.private.get_account().data['account']['positionId']
    percent_slippage = 0.003

    price = float(client.public.get_markets(market = symbol).data['markets'][symbol]['oraclePrice'])

    price_round = str(price)[::-1].find('.')

    price = float(decimal.Decimal(price).quantize(decimal.Decimal(str(1 / pow(10, int(price_round)))), rounding = decimal.ROUND_UP))
    take_profit = float(decimal.Decimal(df['TP_' + side][len(df) - 1]).quantize(decimal.Decimal(str(1 / pow(10, int(price_round)))), rounding = decimal.ROUND_UP))
    stop_loss = float(decimal.Decimal(df['SL_' + side][len(df) - 1]).quantize(decimal.Decimal(str(1 / pow(10, int(price_round)))), rounding = decimal.ROUND_UP))

    tpsl = [take_profit, stop_loss]

    if side == 'BUY':
        price *= (1 + percent_slippage)
        tpsl_side = 'SELL'

    elif side == 'SELL':
        price *= (1 - percent_slippage)
        tpsl_side = 'BUY'

    expiration_epoch_seconds = int((pd.Timestamp.utcnow() + pd.Timedelta(weeks = 5)).timestamp())

    order_params = dict()

    order_params['position_id'] = position_id
    order_params['market'] = symbol
    order_params['side'] = side
    order_params['order_type'] = order_type
    order_params['post_only'] = False
    order_params['size'] = str(size)
    order_params['price'] = str(price)
    order_params['limit_fee'] = '0.0015'
    order_params['expiration_epoch_seconds'] = expiration_epoch_seconds
    order_params['time_in_force'] = 'IOC'
    
    client.private.create_order(**order_params)

    for i in range(2):
        order_params['side'] = tpsl_side
        order_params['order_type'] = TPSL_ORDER_TYPE[i]
        order_params['reduce_only'] = True
        order_params['price'] = str(tpsl[i])
        order_params['trigger_price'] = str(tpsl[i])
        order_params['time_in_force'] = 'IOC'

        client.private.create_order(**order_params)