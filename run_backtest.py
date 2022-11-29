from strategies import Supertrend, fetch_date_highest_price
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from get_trendl import *
from pathlib import Path
import pandas as pd
import progressbar 
import numpy as np
import time




class Trendline_test(Strategy):

    def init(self):

        self.bar = progressbar.ProgressBar(max_value=len(self.data)).start()

        self.last_comb = list()
        self.length = list()
        self.y_price = list()
        self.crossover = False
        self.plotted = True

        self.df = self.data.df


    def next(self):

        self.bar.update(len(self.data))
        
        if  self.plotted and crossover(self.data.Lowerband, self.data.Close):


            self.crossover = True
            self.plotted = False

            self.swing_high = fetch_date_highest_price(self, -20)
    

        if self.crossover:  

            i = len(self.data)
            I = self.data.index[self.swing_high:i].copy()
            C = self.data.Close[self.swing_high:i].copy()
            O = self.data.Open[self.swing_high:i].copy()

            x_peaks = detect_peaks_guassian(C, 0.1)
            if x_peaks is False:
                return

            x_peaks_combinations = all_combi_af_peaks(x_peaks, self.last_comb)
            y_peaks_combinations = fetch_y_values_peaks(C, x_peaks_combinations)
            candidates_df, peak_tup, y_hat= peak_regression(C, x_peaks_combinations, y_peaks_combinations)
            if not candidates_df is None:          
                
                #candidates_df = trendline_angle_degree(candidates_df)
                #candidates_after_check = check_trendl_parameters(candidates_df)
                tup_data_for_plotting  = extract_data_for_plotting(C, I, candidates_df, x_peaks, peak_tup, self.length, self.y_price)

                if tup_data_for_plotting:

                    self.plotted = plot_final_peaks_and_final_trendline(self.df[self.swing_high:i+200].copy(), tup_data_for_plotting, y_hat, I[peak_tup[0]], peak_tup, candidates_df, fit_plot=0)
                
                    if self.plotted == True:
                        print(f'Trendline have been found - {I[peak_tup[0]]}')
                        #time.sleep(2)
                        self.crossover = False
                        self.last_comb.clear()



        self.bar.update(len(self.data))


    
df = pd.read_pickle('./data/ETHUSDT15M.pkl')
df.drop(['Close_time', 'Volume'], axis=1, inplace=True)

# Supertrend
atr_period = 7
atr_multiplier = 2.5
supertrend = Supertrend(df, atr_period, atr_multiplier)
df = df.join(supertrend)

df = df.loc['2022-10-09':]
print(df)

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