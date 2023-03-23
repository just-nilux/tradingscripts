import pandas as pd
import numpy as np
from talipp.indicators import RSI, EMA
from ta import momentum, trend

class TA:

    def rsi(self, candles, window: int):
        rsi = momentum.RSIIndicator(candles["Close"], window)
        # print(f"RSI = {round(rsi.rsi().iloc[-1], 2)}")
        return rsi.rsi()

    def ema(self, candles, window: int):
        ema = trend.EMAIndicator(candles["Close"], window)
        # print(f"EMA = {round(ema.ema_indicator().iloc[-1], 2)}")
        return ema.ema_indicator()

    def sma(self, candles, window: int):
        sma = trend.SMAIndicator(candles["Close"], window) # => rolling(window).mean() on a df
        # print(f"SMA = {round(sma.sma_indicator().iloc[-1], 2)}")
        return sma.sma_indicator()

    def vma(self, candles, window: int):
        vma = candles["Volume"].rolling(window).mean()
        return vma

    def up_candle(self, candles):
        return candles["Close"] > candles["Open"]
    
tas = TA()

def tfs(df: pd.DataFrame):

    RSI_LENGTH = 14
    EMA_LENGTH = 21
    VOLUME_LENGTH = 20

    vol_sma = tas.vma(df, VOLUME_LENGTH)
    up_candle = tas.up_candle(df)

    ema21 = tas.ema(df, EMA_LENGTH)
    rsi14 = tas.rsi(df, RSI_LENGTH)

    # TFS
    cond1 = df["Close"] > ema21
    cond3 = rsi14 > 50
    cond4 = up_candle
    cond5 = df["Volume"] > vol_sma

    # TFSv2
    prev_open = df["Open"].shift(1)
    rsi_below = rsi14 < 50

    v2_cond1 = df["Close"] > prev_open
    v2_cond3 = np.any([rsi_below, rsi_below.shift(1), rsi_below.shift(2)], axis=0)

    df["tfs"] = np.all([cond1, cond3, cond4, cond5, v2_cond1, v2_cond3], axis=0)

    # remove consecutive tfs signals
    cumsum = (df["tfs"] == False).cumsum() # will increment each time tfs == False
    cumcount = df["tfs"].groupby(cumsum).cumcount() # will count in each "cumsum" group
    df["tfs"] = df["tfs"] & (cumcount % 2 == 1)

    return df

