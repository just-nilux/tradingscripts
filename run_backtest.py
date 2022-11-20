from strategies import Supertrend
from backtesting.lib import crossover
from backtesting import Backtest, Strategy
from pathlib import Path
import pandas as pd
import numpy as np
from get_trendl import *
from datetime import timedelta
import time




class Trendline_test(Strategy):

    def init(self):

        self.crossover = False
        self.plotted = True


    def next(self):
        
        if  self.plotted and crossover(self.data.Lowerband, self.data.Close): 

            self.crossover = True
            self.plotted = False

            i = len(self.data)
        
            self.swing_high = len(self.data.index) - self.data.Close[i -30:i].argmax(axis=0)
    

        if self.crossover:  

            i = len(self.data)
            index = self.data.index[self.swing_high:i].copy()
            close = self.data.Close[self.swing_high:i].copy()
            open = self.data.Open[self.swing_high:i].copy()
            s_max = np.maximum(open, close)


            x_peaks = detect_peaks_guassian(index, s_max)
            if x_peaks is False:
                return
            x_peaks_combinations_list = all_combi_af_peaks(x_peaks)
            y_peaks_combination_list = fetch_y_values_peaks(s_max, x_peaks_combinations_list)
            candidates_df, peak_tup = peak_regression(x_peaks_combinations_list, y_peaks_combination_list)
            if not candidates_df is None:          
                
                candidates_df = trendline_angle_degree(candidates_df)
                candidates_after_check = check_trendl_parameters(candidates_df)
                tup_data_for_plotting  = extract_data_for_plotting_numpy(close, index, candidates_after_check, peak_tup)

                if tup_data_for_plotting:
                    
                    self.plotted = plot_final_peaks_and_final_trendline(self.data.df[self.swing_high:i].copy(), tup_data_for_plotting, peak_tup)
                
                    if self.plotted == True:
                        print(f'Trendline have been found - {index[peak_tup[0]]}')
                        self.crossover = False



    
df = pd.read_pickle('./data/ETHUSDT15M.pkl')
df.drop(['Close_time', 'Volume'], axis=1, inplace=True)

# Supertrend
atr_period = 7
atr_multiplier = 2.5
supertrend = Supertrend(df, atr_period, atr_multiplier)
df = df.join(supertrend)

df = df.loc['2022-11-14':]


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