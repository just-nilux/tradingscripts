import json
import pandas_ta as ta
#import xgboost as xgb

with open('config.json') as config_file:
    config = json.load(config_file)

def strategy(df, symbol, strat):
    price = df.Close.astype(float)

    if symbol == 0 and strat == 0:
        df['SMA20'] = df.ta.sma(20)
        df['LONG'] = price == df.SMA20
        df['TP_BUY'] = price * 1.05
        df['SL_BUY'] = price * 0.95
    return df




