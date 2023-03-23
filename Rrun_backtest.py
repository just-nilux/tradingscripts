from strategies import Supertrend, fetch_date_highest_price
from scipy.signal import argrelmax
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from Rget_trendl_3T import *
import pandas as pd
import progressbar 
import numpy as np
import time




class Trendline_test(Strategy):

    def init(self):

        self.bar = progressbar.ProgressBar(max_value=len(self.data)).start()

        self.last_comb = list()
        self.last_comb_arr = np.array([[]])
        self.length = list()
        self.y_price = list()
        self.prev_peak_cnt = 0
        self.tren_df = pd.DataFrame()
        self.df = self.data.df.copy()
        self.prev_plt_idx = None
        self.run = False


    def next(self):

        self.bar.update(len(self.data))


        if crossover(self.data.Close, self.data.Upperband):

            self.run = False
            self.prev_plt_idx = None

                    
        if  crossover(self.data.Lowerband, self.data.Close):

            self.run = True
            self.swing_high = fetch_date_highest_price(self, -40)
    
        
        if self.run:  

            i = len(self.data)

            I = self.data.index[self.swing_high:i].copy()
            C = self.data.Close[self.swing_high:i].copy()
            O = self.data.Open[self.swing_high:i].copy()
            smax = max(C,O)

            x_peaks = argrelmax(smax)[0]
        
            if self.prev_peak_cnt == len(x_peaks) or len(x_peaks) <3:
                return

            self.prev_peak_cnt = len(x_peaks)
            
            #----------------------
            
            #x_peaks_comb = genComb_x3(x_peaks)
            x_peaks_comb = all_combi_af_peaks(x_peaks, self.last_comb)

            #diff = setdiff2d_nb(x_peaks_comb, self.last_comb_arr)
            
            #if len(diff) == 0:
            #    return

            #self.last_comb_arr = x_peaks_comb


            #if diff is None:
            #    return

            y_peaks_comb = fetch_y_values_peaks(smax, x_peaks_comb)
            candidates_df, peak_tup, y_hat= peak_regression(smax, x_peaks_comb, y_peaks_comb, self.tren_df)
            if candidates_df is None:
                return          
                
            tup_data_for_plotting  = extract_data_for_plotting(smax, I, candidates_df, x_peaks, peak_tup, self.length, self.y_price)
            
            self.prev_plt_idx = plot_final_peaks_and_final_trendline(self.df[self.swing_high:].copy(), tup_data_for_plotting, y_hat, I[peak_tup[0]], peak_tup, candidates_df, fit_plot=0)

            self.swing_high = int(np.where(self.data.index == self.prev_plt_idx)[0]) 
            
            self.tren_df = self.tren_df.head(0)
            self.last_comb_arr = np.array([[]])
            self.last_comb.clear()
            self.prev_peak_cnt = 0
            print(f'Trendline have been found - {I[peak_tup[0]]}')




        self.bar.update(len(self.data))


    
df = pd.read_pickle('./data/ETHUSDT1D.pkl')
df.drop(['Close_time', 'Volume'], axis=1, inplace=True)

# Supertrend
atr_period = 7
atr_multiplier = 5
supertrend = Supertrend(df, atr_period, atr_multiplier)
df = df.join(supertrend)

df = df.loc['2022-01-01':'2023-02-01']                            #.loc['2022-11-14 11:15:00' : '2022-11-14 23:15:00']#['2022-10-31':'2022-11-22'] #['2022-10-09':]
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