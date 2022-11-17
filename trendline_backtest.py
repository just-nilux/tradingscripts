from backtesting.lib import crossover
from backtesting import Backtest, Strategy
from backtesting.test import SMA
from pathlib import Path
import pandas as pd
import numpy as np
from get_trendline_backtest_optimized import *
from datetime import timedelta


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


        self.idxmax = self.data.index[-1]

        self.crossover = False
        self.plotted = True

        self.df = self.data.df.copy()
        self.df.drop(['High', 'Low', 'Close_time', 'Supertrend', 'Lowerband', 'Upperband', 'Volume'], axis=1, inplace=True)


    def next(self):
        
    
        if self.plotted and crossover(self.data.Lowerband, self.data.Close): 
            self.idxmax = self.data.Close.s.tail(30).idxmax() 
            self.crossover = True
            self.plotted = False


        if self.crossover:  

            current_id = self.data.index[-1]

            df = self.df.loc[self.idxmax:current_id].copy()

            x_peaks = detect_peaks_guassian(df)
            x_peaks_combinations_list = all_combi_af_peaks(x_peaks)
            y_peaks_combination_list = fetch_y_values_peaks(df, x_peaks_combinations_list)
            trendl_candidates_df = peak_regression(x_peaks_combinations_list, y_peaks_combination_list)
            if not trendl_candidates_df.empty:          
                trendl_candidates_df = fetch_trendl_start_end_price(df, trendl_candidates_df)
                trendl_candidates_df = trendline_angle_degree(trendl_candidates_df)
                candidates_after_check = check_trendl_parameters(trendl_candidates_df)
                tup_data_for_plotting  = extract_data_for_plotting(df, candidates_after_check, x_peaks)

                
                if tup_data_for_plotting:
                    df_plot_id = current_id + timedelta(days=3)
                    df_plot = self.data.df.loc[self.idxmax:df_plot_id]
                    
                    self.plotted = plot_final_peaks_and_final_trendline(df_plot, tup_data_for_plotting, x_peaks)

                #self.plotted = test_feed(df)

                    if self.plotted == True:
                        print('Trendline have been found')
                        self.crossover = False


            # 1 - Fetch idxmax for starting point of peak search 
            # 2 - Feed df with same start index to "detect_peaks_gaussion, add one candle more at a time in each iteration.  
            # 3 - contuine untill there is a minimum of 4 peaks returned from detect_peaks()
            # 4 - Run all methods of get_trend_line.py and wait untill r_value == 0.999
            # 5 - if peaks is found and regression with r_value == 0.999 -> plot / if none have been found before next crossover do nothing, and wait for next crossover
            # 6 - start process over again...


    
df = pd.read_pickle('ETHUSDT15M.pkl')#.loc['2022-10':]#.loc['2018-05-06':'2018-05-07 03:00:00'] #.loc['2022-11-09':'2022-11-10']

# Supertrend
atr_period = 30
atr_multiplier = 5
supertrend = Supertrend(df, atr_period, atr_multiplier)
df = df.join(supertrend)

df = df.loc['2022-10']

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