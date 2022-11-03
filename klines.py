from datetime import datetime, timedelta
from binance.client import Client
import datetime, time
import pandas as pd
import os


def formatDataframe(data):

    df = pd.DataFrame(
            data, 
            columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
    )
    
    date = pd.to_datetime(df['Date'],unit='ms').dt.strftime("%Y-%m-%d %H:%M:%S")
    print(type(date))
    df['Close_time'] = pd.to_datetime(df['Close_time'],unit='ms') + timedelta(seconds=1)
    Close_time = df['Close_time'].dt.strftime('%Y-%m-%d %H:%M:%S').astype(str).str.upper()
    
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    df = df.join(date)
    df = df.join(Close_time)
    
    df.set_index('Date', inplace=True)
    df = df.loc[~df.index.duplicated()]
    df = df.sort_index()

    return df


def klineHunter(symbol, interval, start=None, stop=None):

    """Method to fetch historical data from Binance API"""

    client = Client(
            "wca05r3DNe36Q3yusf3uJlpyW7qfZGYP623DtrgeHynzfWct6Kv5jINCxZF684rd", 
            "28DE1cQbb7427tlVP4ZBqhkR0etHr9ErSxVd4htIngHWN8y5ZT6WjYz87aint39u"
    )

    data = client.get_historical_klines(
            str(symbol).upper(), 
            interval, 
            start, 
            stop
    )

    formatted_df = formatDataframe(data)

    path = '//home/traderblakeq/Python/klines'
    os.chdir(path)

    formatted_df.to_pickle(f'{symbol.upper()}{interval.upper()}.pkl')

    print(f'Historical data have been saved to: {path}/{symbol.upper()}{interval.upper()}.pkl')
    
    return formatted_df



