
from binance.client import BinanceAPIException
from datetime import datetime, timedelta
from binance.client import Client
from tqdm import tqdm
import datetime, time
import pandas as pd
import requests
import time
import os
import re


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
    print(df.head(1))
    
    return df


def main(symbol, interval, start=None, stop=None, path=None,*args, **kwargs):

    """Method to fetch historical data from Binance API  - ex.: df = klineHunter('ethusdt', '1d', '2022-01-01')"""

    client = Client("wca05r3DNe36Q3yusf3uJlpyW7qfZGYP623DtrgeHynzfWct6Kv5jINCxZF684rd", "28DE1cQbb7427tlVP4ZBqhkR0etHr9ErSxVd4htIngHWN8y5ZT6WjYz87aint39u")

    #print(path)
    data = client.get_historical_klines(str(symbol).upper(), interval, start, stop)
    
    df = formatData(data)
    
    os.chdir(path)

    df.to_pickle(f'{symbol.upper()}{interval.upper()}.pkl')
    print(f'Historical data have been saved to: {path}/{symbol.upper()}{interval.upper()}.pkl')    
    return df


def get_top_50_volume_pair_symbols():
    url = 'https://fapi.binance.com/fapi/v1/ticker/24hr'
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        
        # Sort by trading volume in descending order
        sorted_data = sorted(data, key=lambda x: float(x['quoteVolume']), reverse=True)
        
        # Get the top 50 trading pairs
        top_50_pairs = [pair['symbol'] for pair in sorted_data[:50]]
        print(top_50_pairs)
    else:
        print(f"Error: {response.status_code}")

    return top_50_pairs


def get_most_recent_file(base_path):
    files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith(".pkl")]
    if not files:
        return None
    
    most_recent_file = max(files, key=os.path.getmtime)
    return os.path.basename(most_recent_file)



def fecth_top_50_volume_pair(base_path=None):

    if base_path is None:
        base_path = '//home/traderblakeq/Python/klines/all_data_old'
    retry_delay = 60  # Time to wait before retrying (in seconds)

    top_50_pairs = get_top_50_volume_pair_symbols()
    intervals = ["5M", "15M", "1H", "4H", "1D", "1W"]
    
    # Get the most recent .pkl file
    most_recent_file = get_most_recent_file(base_path)

    if most_recent_file is not None:
        # Extract the symbol and interval from the most recent file
        pattern = r"(\w+)(\d+[a-zA-Z]+)\.pkl"
        match = re.match(pattern, most_recent_file)
        if match:
            last_downloaded_symbol, last_downloaded_interval = match.groups()
        else:
            last_downloaded_symbol = None
            last_downloaded_interval = None

        last_downloaded_symbol, last_downloaded_interval = match.groups()
        
        # Find the indices of the last downloaded symbol and interval
        last_symbol_index = top_50_pairs.index(last_downloaded_symbol)
        last_interval_index = intervals.index(last_downloaded_interval)
    else:
        last_symbol_index = 0
        last_interval_index = 0

    for symbol_index in tqdm(range(last_symbol_index, len(top_50_pairs)), desc="Symbols"):
        symbol = top_50_pairs[symbol_index]
        for interval_index in range(last_interval_index, len(intervals)):
            interval = intervals[interval_index]

            existing_file = f"{symbol.upper()}_{interval.upper()}.pkl"
            if existing_file in os.listdir(base_path):
                print(f"Data for {symbol.upper()}_{interval.upper()} already exists. Skipping...")
                continue

            while True:
                try:
                    print(f"Downloading data for {symbol.upper()}_{interval.upper()}...")
                    main(symbol, interval.lower(), start="2016", path=base_path)
                    print(f"Data for {symbol.upper()}_{interval.upper()} downloaded.")
                    break  # If the download is successful, break the loop and continue with the next interval
                except BinanceAPIException as e:
                    if e.code == -1121:
                        print(f"Invalid symbol: {symbol.upper()}_{interval.upper()}. Skipping...")
                        break  # Skip the symbol if the API error is "Invalid symbol"
                    else:
                        print(f"Error downloading data for {symbol.upper()}_{interval.upper()}: {e}")
                        print(f"Waiting {retry_delay} seconds before retrying...")
                        time.sleep(retry_delay)
                except Exception as e:
                    print(f"Error downloading data for {symbol.upper()}_{interval.upper()}: {e}")
                    print(f"Waiting {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)

        # Reset the interval index back to 0 for the next symbol
        last_interval_index = 0


    else:
        print("All datasets have been downloaded.")


fecth_top_50_volume_pair()