from datetime import datetime, timedelta
from binance.client import Client
import datetime, time
import pandas as pd
import os


def formatData(data):

    df = pd.DataFrame(data, columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']).astype(float)
    df.drop(columns=['Date', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'], inplace=True)

    #df['Date'] = pd.to_datetime(df['Date'],unit='ms').apply(lambda x: x.to_datetime64())
    df['Date'] = pd.to_datetime(df['Close_time'],unit='ms') + timedelta(seconds=0.001)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['Date'] = pd.to_datetime(df['Date']).apply(lambda x: x.to_datetime64())

    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    df = df.loc[~df.index.duplicated()]
    del df['Close_time']
    print(df)
    
    return df


def main(symbol, interval, start=None, stop=None, path=None,*args, **kwargs):

    """Method to fetch historical data from Binance API  - ex.: df = klineHunter('ethusdt', '1d', '2022-01-01')"""

    client = Client("wca05r3DNe36Q3yusf3uJlpyW7qfZGYP623DtrgeHynzfWct6Kv5jINCxZF684rd", "28DE1cQbb7427tlVP4ZBqhkR0etHr9ErSxVd4htIngHWN8y5ZT6WjYz87aint39u")

    #print(path)
    data = client.get_historical_klines(str(symbol).upper(), interval, start, stop)
    
    df = formatData(data)
    if path == None:
        path = '//home/traderblakeq/Python/klines'
        os.chdir(path)

        df.to_pickle(f'{symbol.upper()}{interval.upper()}.pkl')
        print(f'Historical data have been saved to: {path}/{symbol.upper()}{interval.upper()}.pkl')

    elif path == 0:
        df.to_pickle(f'{symbol.upper()}{interval.upper()}.pkl')
        print(f'Historical data have been saved to: {os.getcwd()}/{symbol.upper()}{interval.upper()}.pkl')


    
    return df



if __name__=='__main__':
    import sys 
    
    if len(sys.argv) == 3:
        symbol, interval = sys.argv[1:3]
        main(symbol, interval)
        print(main.__doc__)

    elif len(sys.argv) == 4:
        symbol, interval, start = sys.argv[1:4]
        main(symbol, interval, start)

    else:
        symbol, interval, start, stop = sys.argv[1:]
        main(symbol, interval, start, stop)