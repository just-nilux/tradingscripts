import json
import datetime
import numpy as np
import pandas as pd
from time import time
from pybit import usdt_perpetual

with open('config.json') as config_file:
    config = json.load(config_file)

client = usdt_perpetual.HTTP(api_key = config['bybit_api_key'], api_secret = config['bybit_api_secret'])

def get_all_symbols():
    return client.query_symbol()['result']

def get_wallet_balance():
    return float(client.get_wallet_balance()['result'][config['base_asset']]['available_balance'])

def get_klines(symbol: str, interval: str):
    raw_candles = client.query_kline(symbol = symbol, interval = config['BYBIT_TIME_FRAMES'][interval], limit = 200, from_time = int(time() - 200 * 60 * config['BYBIT_TIME_FRAMES'][interval]))['result']

    open = np.array([])
    close = np.array([])
    high = np.array([])
    low = np.array([])
    volume = np.array([])
    date = np.array([])

    for i in range(len(raw_candles)):
        open = np.append(open, raw_candles[i]['open'], axis = None)
        close = np.append(close, raw_candles[i]['close'], axis = None)
        high = np.append(high, raw_candles[i]['high'], axis = None)
        low = np.append(low, raw_candles[i]['low'], axis = None)
        volume = np.append(volume, raw_candles[i]['volume'], axis = None)
        timestamp = int(raw_candles[i]['open_time'])
        value = datetime.datetime.fromtimestamp(timestamp)
        date = np.append(date, f"{value:%Y-%m-%d %H:%M}", axis = None)

    return pd.DataFrame({'Open': open, 'High': high, 'Low': low, 'Close': close, 'Volume': volume}, index = date).rename_axis('DateTime')

def place_order(symbol: str, size: float, side: str, order_type: str, take_profit_price: float, stop_loss_price: float):
    client.place_active_order(symbol = symbol, qty = size, side = side, order_type = order_type, time_in_force = 'GoodTillCancel', reduce_only = False, close_on_trigger = False, take_profit = take_profit_price, stop_loss = stop_loss_price)

def change_leverage(symbol: str, leverage: int):
    client.set_leverage(symbol = symbol, buy_leverage = leverage, sell_leverage = leverage)

def position_check(symbol: str):
    response = client.my_position(symbol = symbol)
    
    if response is not None:
        if not((float(response['result'][0]['entry_price']) == 0) or (float(response['result'][1]['entry_price']) == 0)):
            return 1

        else:
            return 0

def position_check_multi():
    in_trade = 0

    for symbol in config['symbols']:
        response = client.my_position(symbol = symbol)
        
        if response is not None:
            if not((float(response['result'][0]['entry_price']) == 0) or (float(response['result'][1]['entry_price']) == 0)):
                in_trade += 1

    return in_trade