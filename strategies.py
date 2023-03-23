import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
import zigzag as zig
import warnings

warnings.filterwarnings('ignore')

def scan_from_ts_to_ts(df, timestamps_l):
    
    # Initialize an empty list to store the timestamps where bearish engulfing candles were found
    timestamps_list = []

    # Loop through each time period in the list
    for start_time, end_time in timestamps_l:
        # Slice the DataFrame to only include data within the current time period
        period_df = df.loc[start_time:end_time]
        
        # Use the detect_bearish_engulfing function to find bearish engulfing candles in the current time period
        #signals = detect_bearish_engulfing(period_df, include_mavol=True, vol_ma_period=20, vol_threshold=1)
        signals = shooting_star(period_df, include_mavol=True, vol_ma_period=20, vol_threshold=1)

        
        # If any bearish engulfing candles were found, append the timestamps to the list
        if signals.any():
            timestamps_list.extend(signals.index[signals].tolist())
    
    # Convert to pd.Series
    timestamps_series = pd.Series(data=True, index=timestamps_list, name='bearish_engulfing')
    
    return timestamps_series



def detect_x_percent_move(df, signals, change=0.03):
    """
    Detects the percentage of times a sample of data, filtered by a specific boolean condition, 
    is followed by a price change of at least a specified percentage, in either direction.

    The function takes a pandas DataFrame `df`, which contains the OHLC data for the asset. 
    The second parameter is a pandas Series `signals`, which contains boolean values indicating the presence of the condition to be filtered. 
    The optional `change` parameter specifies the minimum percentage change that is to be detected.

    The function returns a pandas DataFrame containing the start and end dates of the detected patterns, 
    along with the percentage change in price, if any. If no patterns are detected, an empty DataFrame is returned. 
    Additionally, the function prints the number of samples in the data and the percentage of cases where the price first rose or fell by the specified percentage.

    """
    
    # Filter out True condition from signals:
    sample_indices = [i for i in signals.index if signals[i] == True]
    sample_indices = df.index.get_indexer_for(sample_indices)
    
    results = []
    
    for i in sample_indices:
        # Look at the next x candles
        for j in range(i+1, len(df)):
            if df['Close'][j] > df['Open'][j]:
                pct_change = (df['High'][j] - df['Close'][i]) / df['Close'][i]            
            elif df['Close'][j] < df['Open'][j]:
                pct_change = (df['Low'][j] - df['Close'][i]) / df['Close'][i]
            if abs(pct_change) >= change:
                results.append((df.index[i], df.index[j], pct_change))
                break
    if not results:
         return pd.DataFrame()
    df = pd.DataFrame(results, columns=['start_date', 'end_date', 'pct_change'])
    
    no_positive = df[df['pct_change']>0].count(axis=0)['pct_change']
    no_negative = df[df['pct_change']<0].count(axis=0)['pct_change']

    total = no_positive + no_negative
    percent_positive = no_positive / total * 100
    percent_negative = no_negative / total * 100
    
    print(f'No. of samples: {len(sample_indices)}')
    print(f"Price first rose {change*100}% in {percent_positive:.2f}% of cases.")
    print(f"Price first fell {change*100}% in {percent_negative:.2f}% of cases.")
          
    return df


def detect_bearish_engulfing(df, n, include_mavol=False, vol_ma_period=20, vol_threshold=1, size_threshold=0.3):
    # Compute the previous day's OHLC values
    prev_close = df['Close'].shift(1)
    prev_open = df['Open'].shift(1)
    prev_high = df['High'].shift(1)
    prev_body_size = abs(prev_close - prev_open)

    # Compute a boolean array indicating whether each candle is a bearish engulfing candle
    current_close = df['Close']
    current_open = df['Open']
    current_high = df['High']
    current_body_size = abs(current_close - current_open)
    prev_n_high = df['High'].rolling(n).max().shift(1)
    is_bearish_engulfing = ((prev_close > prev_open) & (current_open > prev_close) & (current_close < prev_open) & (current_body_size / prev_body_size > (1 + size_threshold)) & (prev_high == prev_n_high))

    if include_mavol:
        # Calculate the moving average of the volume
        df['vol_ma'] = df['Volume'].rolling(vol_ma_period).mean()

        # Determine which candlesticks meet the volume criteria (volume > x * volume moving average)
        is_high_vol = (df['Volume'] > vol_threshold * df['vol_ma'])

        # Combine the shooting star and volume criteria to get the final set of signals
        signals = is_bearish_engulfing & is_high_vol

        return signals
    else:
        return is_bearish_engulfing



def shooting_star(df, n, include_mavol=False, vol_ma_period=20, vol_threshold=1):
    
    # Calculate the body, upper shadow, and lower shadow of each candlestick
    df['body'] = abs(df['Open'] - df['Close'])
    df['upper_shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    prev_high = df['High'].shift(1)
    
    # Determine which candlesticks meet the shooting star criteria (body <= 30% of full size, lower shadow <= 5% of full size)
    is_shooting_star = (df['body'] <= 0.25 * (df['High'] - df['Low'])) & (df['lower_shadow'] <= 0.05 * (df['High'] - df['Low'])) & (df['Close'] < prev_high) & (df['High'] == df['High'].rolling(window=n).max())
    
    if include_mavol:

        # Calculate the moving average of the volume
        df['vol_ma'] = df['Volume'].rolling(vol_ma_period).mean()
        
        # Determine which candlesticks meet the volume criteria (volume > x * volume moving average)
        is_high_vol = (df['Volume'] > vol_threshold * df['vol_ma'])
        
        # Combine the shooting star, highest in n periods, and volume criteria to get the final set of signals
        signals = is_shooting_star & is_high_vol
    
        return signals
    else:
        return is_shooting_star

def detect_red_sell_candle(df, n, include_mavol=False, vol_ma_period=20, vol_threshold=1):
        prev_n_high = df['High'].rolling(n).max().shift(1)

        is_red_sell_candle = (df['Close'] < df['Open']) & (df['High'] == df['High'].rolling(window=n).max())


        if include_mavol:
            # Calculate the moving average of the volume
            df['vol_ma'] = df['Volume'].rolling(vol_ma_period).mean()

            # Determine which candlesticks meet the volume criteria (volume > x * volume moving average)
            is_high_vol = (df['Volume'] > vol_threshold * df['vol_ma'])

            # Combine the shooting star and volume criteria to get the final set of signals
            signals = is_red_sell_candle & is_high_vol

            return signals
        
        else:
            return is_red_sell_candle



def fair_value_gap_detector(df, threshold=0.02):
    """
    Calculates and appends a 'Fair Value Gap' column to a given pandas dataframe, 
    indicating whether a security is over or undervalued based on its recent trading data.  

    Parameters:
    df (pandas.DataFrame): The input dataframe containing the relevant trading data.
    threshold (float): The minimum threshold for determining if a security is over 
    or undervalued (default 0.02).

    Returns:
    pandas.DataFrame: The original dataframe with an added 'Fair Value Gap' column 
    indicating the over or undervaluation status of a security.
    """
    
    fair_value_gap = [0,0]
    for i in range(2, len(df)):

        if (df.iloc[i]['Low'] > df.iloc[i-2]['High']) & (df.iloc[i-1]['Close'] > df.iloc[i-1]['Open'] + df.iloc[i-1]['Open'] * threshold) & (df.iloc[i-1]['Close'] > df.iloc[i-2]['High']):
            fair_value_gap.append(1)

        elif (df.iloc[i]['High'] < df.iloc[i-2]['Low']) & (df.iloc[i-1]['Close'] < df.iloc[i-1]['Open'] - df.iloc[i-1]['Open'] * threshold) & (df.iloc[i-1]['Close'] < df.iloc[i-2]['Low']):
            fair_value_gap.append(-1)
        
        else:
            fair_value_gap.append(0)
    
    
    df['FVG'] = fair_value_gap

    return df



def plot_zigzag(X, pivots, struc, consecutive):
    """ 
        X: df.Close
        pivots: df.zigzag
        struc: df.struc 
        consecutive: df.consecutive
    """
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(X)), X, 'k:', alpha=0.5)
    ax.plot(np.arange(len(X))[pivots != 0], X[pivots != 0], 'k-')
    ax.scatter(np.arange(len(X))[pivots == 1], X[pivots == 1], color='g')
    ax.scatter(np.arange(len(X))[pivots == -1], X[pivots == -1], color='r')
    if struc != None:
        for (x,y) in zip(np.arange(len(X))[pivots == -1], X[pivots == -1]):
            ax.annotate(struc[x], (x, y))
            #ax.annotate(consecutive[x], (x+1, y))
        for (x,y) in zip(np.arange(len(X))[pivots == 1], X[pivots == 1]): 
            ax.annotate(struc[x], (x, y))
            #ax.annotate(consecutive[x], (x+1, y))


    plt.show()

def plot_zig(X, pivots):
    """ 
        X: df.Close
        pivots: df.zigzag
        
    """
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(X)), X, 'k:', alpha=0.5)
    ax.plot(np.arange(len(X))[pivots != 0], X[pivots != 0], 'k-')
    ax.scatter(np.arange(len(X))[pivots == 1], X[pivots == 1], color='g')
    ax.scatter(np.arange(len(X))[pivots == -1], X[pivots == -1], color='r')
    plt.show()



def consecutive_detect(formations):

    for i in range(len(formations), 0, -1):
        if len(set(formations[-i:])) == 1:
            return i
    return 1


def zigzag_old(df, repaint=True):
    
    df.reset_index(inplace=True)

    pivots = zig.peak_valley_pivots(df.Close, 0.04, -0.04)

    if not repaint:
        pivots[-1] = 0

    df['zigzag'] = pivots
        
    up = df.loc[df['zigzag'] == 1]
    down = df.loc[df['zigzag'] ==-1]

    tmp_list_up = list()
    close_up = up.Close.values.tolist()
    for i, row in enumerate(up.itertuples()):
        if i == 0:
            
            if row[5] < close_up[+1]:
                up.loc[row[0], 'struc'] = 'LH'
                tmp_list_up.append('LH')
            
            elif row[5] > close_up[+1]:
                up.loc[row[0], 'struc'] = 'HH'
                tmp_list_up.append('HH')
        
        elif row[5] < close_up[i-1]:
            up.loc[row[0], 'struc'] = 'LH'
            tmp_list_up.append('LH')
        
        elif row[5] > close_up[i-1]:
            up.loc[row[0], 'struc'] = 'HH'
            tmp_list_up.append('HH')
        up.loc[row[0], 'consecutive'] = consecutive_detect(tmp_list_up)



    tmp_list_down = list()
    close_down = down.Close.values.tolist()
    for i, row in enumerate(down.itertuples()):
        if i == 0:
            if row[5] < close_down[i+1]:
                down.loc[row[0], 'struc'] = 'LL'
                tmp_list_down.append('LL')

            elif row[5] > close_down[i+1]:
                down.loc[row[0], 'struc'] = 'HL'
                tmp_list_down.append('HL')
    
        elif row[5] < close_down[i-1]:
            down.loc[row[0], 'struc'] = 'LL'
            tmp_list_down.append('LL')

        elif row[5] > close_down[i-1]:
            down.loc[row[0], 'struc'] = 'HL'
            tmp_list_down.append('HL')
        down.loc[row[0], 'consecutive'] = consecutive_detect(tmp_list_down)

    up_down = pd.concat([up,down], axis=0)
    res = pd.concat([df, up_down['struc'], up_down['consecutive']], axis=1)

    res.set_index('Date', inplace=True)

    return pd.DataFrame(res[['zigzag', 'struc', 'consecutive']])                      


def zigzag(df, repaint=True):

    #if len(df)<100:
    #    return None
    df.reset_index(inplace=True)

    pivots = zig.peak_valley_pivots(df.Close, 0.01, -0.01)

    if not repaint:
        pivots[-1] = 0

    df['zigzag'] = pivots
        
    up = df.loc[df['zigzag'] == 1]
    down = df.loc[df['zigzag'] ==-1]

    tmp_list_up = []
    close_up = up.Close.values.tolist()
    for i, row in enumerate(up.itertuples()):
        if i == 0:
            if row[5] < close_up[i+1]:
                struc = 'LH'
            elif row[5] > close_up[i+1]:
                struc = 'HH'
        elif row[5] < close_up[i-1]:
            struc = 'LH'
        elif row[5] > close_up[i-1]:
            struc = 'HH'
        tmp_list_up.append(struc)
        up.loc[row[0], 'struc'] = struc
        up.loc[row[0], 'consecutive'] = consecutive_detect(tmp_list_up)


    tmp_list_down = []
    close_down = down.Close.values.tolist()
    for i, row in enumerate(down.itertuples()):
        if i == 0:
            if row[5] < close_down[i+1]:
                struc = 'LL'
            elif row[5] > close_down[i+1]:
                struc = 'HL'
        elif row[5] < close_down[i-1]:
            struc = 'LL'
        elif row[5] > close_down[i-1]:
            struc = 'HL'
        tmp_list_down.append(struc)
        down.loc[row[0], 'struc'] = struc
        down.loc[row[0], 'consecutive'] = consecutive_detect(tmp_list_down)

    up_down = pd.concat([up,down], axis=0)
    res = pd.concat([df, up_down[['struc', 'consecutive']]], axis=1)

    res.set_index('Date', inplace=True)

    return pd.DataFrame(res[['zigzag', 'struc', 'consecutive']])

def zigzag_rsi(df, repaint=True):

    if len(df)<100:
        return None
    df.reset_index(inplace=True)

    col_idx = df.columns.get_loc("rsi")

    pivots = zig.peak_valley_pivots(df.rsi, 0.03, -0.03)

    if not repaint:
        pivots[-1] = 0

    df['zigzag_rsi'] = pivots
        
    up = df.loc[df['zigzag_rsi'] == 1]
    down = df.loc[df['zigzag_rsi'] ==-1]

    tmp_list_up = []
    close_up = up.rsi.values.tolist()
    for i, row in enumerate(up.itertuples()):
        if i == 0:
            if row[col_idx+1] < close_up[i+1]:
                struc = 'LH'
            elif row[col_idx+1] > close_up[i+1]:
                struc = 'HH'
        elif row[col_idx+1] < close_up[i-1]:
            struc = 'LH'
        elif row[col_idx+1] > close_up[i-1]:
            struc = 'HH'
        tmp_list_up.append(struc)
        up.loc[row[0], 'struc_rsi'] = struc
        up.loc[row[0], 'consecutive_rsi'] = consecutive_detect(tmp_list_up)


    tmp_list_down = []
    close_down = down.rsi.values.tolist()
    for i, row in enumerate(down.itertuples()):
        if i == 0:
            if row[col_idx+1] < close_down[i+1]:
                struc = 'LL'
            elif row[col_idx+1] > close_down[i+1]:
                struc = 'HL'
        elif row[col_idx+1] < close_down[i-1]:
            struc = 'LL'
        elif row[col_idx+1] > close_down[i-1]:
            struc = 'HL'
        tmp_list_down.append(struc)
        down.loc[row[0], 'struc_rsi'] = struc
        down.loc[row[0], 'consecutive_rsi'] = consecutive_detect(tmp_list_down)

    up_down = pd.concat([up,down], axis=0)
    res = pd.concat([df, up_down[['struc_rsi', 'consecutive_rsi']]], axis=1)

    res.set_index('Date', inplace=True)

    return pd.DataFrame(res[['zigzag_rsi', 'struc_rsi', 'consecutive_rsi']])


def MACDV(df):
    # MACD-V Calc: (( 12 bar EMA - 26 bar EMA) / ATR(26)) * 100
    # https://www.naaim.org/wp-content/uploads/2022/05/MACD-V-Alex-Spiroglou-WEB.pdf

    high = df['High']
    low = df['Low']
    close = df['Close']
    
    fast_ema = ta.ema(close, length=12, talib=None, offset=None)
    slow_ema = ta.ema(close, length=26, talib=None, offset=None)
    atr = ta.atr(high, low, close, length=26, mamode=None, talib=True, drift=None, offset=None)
    
    macdv = ((fast_ema - slow_ema) / atr) * 100
    signal_line = ta.ema(macdv, length=9, talib=None, offset=None)
    hist = macdv - signal_line
    
    return pd.DataFrame({'macdv':macdv, 'signal':signal_line, 'hist':hist})



def Supertrend(df, atr_period, multiplier):
    
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # calculate ATR
    price_diffs = [high - low, 
                   high - close.shift(), 
                   close.shift() - low]
    true_range = pd.concat(price_diffs, axis=1)
    true_range = true_range.abs().max(axis=1)
    # default ATR calculation in supertrend indicator
    atr = true_range.ewm(alpha=1/atr_period,min_periods=atr_period).mean() 
    # df['atr'] = df['tr'].rolling(atr_period).mean()
    
    # HL2 is simply the average of high and low prices
    hl2 = (high + low) / 2
    # upperband and lowerband calculation
    # notice that final bands are set to be equal to the respective bands
    final_upperband = upperband = hl2 + (multiplier * atr)
    final_lowerband = lowerband = hl2 - (multiplier * atr)
    
    # initialize Supertrend column to True
    supertrend = [True] * len(df)
    
    for i in range(1, len(df.index)):
        curr, prev = i, i-1
        
        # if current close price crosses above upperband
        if close[curr] > final_upperband[prev]:
            supertrend[curr] = True
        # if current close price crosses below lowerband
        elif close[curr] < final_lowerband[prev]:
            supertrend[curr] = False
        # else, the trend continues
        else:
            supertrend[curr] = supertrend[prev]
            
            # adjustment to the final bands
            if supertrend[curr] == True and final_lowerband[curr] < final_lowerband[prev]:
                final_lowerband[curr] = final_lowerband[prev]
            if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
                final_upperband[curr] = final_upperband[prev]
    
    return pd.DataFrame({
        'Supertrend': supertrend,
        'Lowerband': final_lowerband.shift(1),
        'Upperband': final_upperband.shift(1)
    }, index=df.index)


def heikin_ashi(df):    
    ha_close = (df['Open'] + df['Close'] + df['High'] + df['Low']) / 4
    
    ha_open = [(df['Open'].iloc[0] + df['Close'].iloc[0]) / 2]
    for close in ha_close[:-1]:
        ha_open.append((ha_open[-1] + close) / 2)    
    ha_open = np.array(ha_open)

    elements = df['High'], df['Low'], ha_open, ha_close
    ha_high, ha_low = np.vstack(elements).max(axis=0), np.vstack(elements).min(axis=0)
    
    return pd.DataFrame({
        'Open': ha_open,
        'High': ha_high,    
        'Low': ha_low,
        'Close': ha_close
    }) 

def logscale(df):

    df.Open = np.log(df['Open'])
    df.Close = np.log(df['High'])
    df.Low = np.log(df['Low'])
    df.Close  = np.log(df['Close'])

    return df

def fetch_date_highest_price(self, n):
            
    indices  = np.arange(len(self.data))

    sliced = self.data.Close[-abs(n):]            
    sliced_idx = indices[-abs(n):]
            
    swing_high = sliced_idx[sliced.argmax(axis=0)]

    return swing_high