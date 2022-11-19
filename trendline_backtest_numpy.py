from backtesting.lib import crossover
from backtesting import Backtest, Strategy
from backtesting.test import SMA
from pathlib import Path
import pandas as pd
import numpy as np
from get_trendline_backtest_numpy import *
from datetime import timedelta


import time




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

# --------------------------------------------------------------------------------------------------------------

class Trendline_test(Strategy):

    def init(self):

        self.swing_high = self.data.index[-1]

        self.crossover = False
        self.plotted = True

        self.trendl_candidates_df = pd.DataFrame(columns =['df_start_index', 'df_end_index', 'slope', 'intercept', 'r_value', 'p_value', 'std_err'])


    def next(self):
        
        if  self.plotted and crossover(self.data.Lowerband, self.data.Close): 

            self.crossover = True
            self.plotted = False

            i = len(self.data)
        
            self.highest_close = len(self.data.index) - self.data.Close[i -30:i].argmax(axis=0)
    

        if self.crossover:  

            i = len(self.data)

            index = self.data.index[self.highest_close:i].copy()
            close = self.data.Close[self.highest_close:i].copy()
            open = self.data.Open[self.highest_close:i].copy()
            s_max = np.maximum(open, close)

            x_peaks = detect_peaks_guassian(index, s_max)
            x_peaks_combinations_list = all_combi_af_peaks(x_peaks)
            y_peaks_combination_list = fetch_y_values_peaks(open, close, x_peaks_combinations_list)
            trendl_candidates_df = peak_regression(self.trendl_candidates_df, x_peaks_combinations_list, y_peaks_combination_list)
            if not trendl_candidates_df.empty:          
                #trendl_candidates_df = fetch_trendl_start_end_price(self.data.df, trendl_candidates_df)
                trendl_candidates_df = trendline_angle_degree(trendl_candidates_df)
                candidates_after_check = check_trendl_parameters(trendl_candidates_df)
                tup_data_for_plotting  = extract_data_for_plotting_numpy(close, index, candidates_after_check, x_peaks)

                if tup_data_for_plotting:

                    self.plotted = plot_final_peaks_and_final_trendline(self.data.df[self.highest_close:i].copy(), tup_data_for_plotting, x_peaks)
                
                    if self.plotted == True:
                        print(f'Trendline have been found - {index[-1]}')
                        self.crossover = False

            # 1 - Fetch idxmax for starting point of peak search 
            # 2 - Feed df with same start index to "detect_peaks_gaussion, add one candle more at a time in each iteration.  
            # 3 - contuine untill there is a minimum of 4 peaks returned from detect_peaks()
            # 4 - Run all methods of get_trend_line.py and wait untill r_value == 0.999
            # 5 - if peaks is found and regression with r_value == 0.999 -> plot / if none have been found before next crossover do nothing, and wait for next crossover
            # 6 - start process over again...


    
df = pd.read_pickle('ETHUSDT15M.pkl')
df.drop(['Close_time', 'Volume'], axis=1, inplace=True)

# Supertrend
atr_period = 7
atr_multiplier = 2.5
supertrend = Supertrend(df, atr_period, atr_multiplier)
df = df.join(supertrend)

df = df.loc['2022-11 - 02:00:00':]


bt = Backtest(df, Trendline_test,
              cash=1000_000, 
              commission=.002,
              exclusive_orders=True,
              #trade_on_close=True,
              hedging=False
              )


output = bt.run()
#print(output)
#bt.plot()