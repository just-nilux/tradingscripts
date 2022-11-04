from datetime import datetime, timedelta
from binance.client import Client
import datetime, time
import pandas as pd
import os


def formatData(data):

    df = pd.DataFrame(data, columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time']].astype(float)

    df['Date'] = pd.to_datetime(df['Date'],unit='ms').dt.strftime("%Y-%m-%d %H:%M:%S")
    df['Close_time'] = pd.to_datetime(df['Close_time'],unit='ms') + timedelta(seconds=1)
    df['Close_time'] = df['Close_time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    df = df.loc[~df.index.duplicated()]
    
    return df


def klineHunter(symbol, interval, start=None, stop=None):

    """Method to fetch historical data from Binance API  - ex.: df = klineHunter('ethusdt', '1d', '2022-01-01')"""

    client = Client("wca05r3DNe36Q3yusf3uJlpyW7qfZGYP623DtrgeHynzfWct6Kv5jINCxZF684rd", "28DE1cQbb7427tlVP4ZBqhkR0etHr9ErSxVd4htIngHWN8y5ZT6WjYz87aint39u")

    data = client.get_historical_klines(str(symbol).upper(), interval, start, stop)

    df = formatData(data)

    path = '//home/traderblakeq/Python/klines'
    os.chdir(path)

    df.to_pickle(f'{symbol.upper()}{interval.upper()}.pkl')

    print(f'Historical data have been saved to: {path}/{symbol.upper()}{interval.upper()}.pkl')
    
    return df



