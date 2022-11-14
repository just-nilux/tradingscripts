from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks_cwt
from scipy.signal import find_peaks
from scipy.stats import linregress
from scipy import signal

from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as fplt
import pandas as pd
import numpy as np
import math
import os

from itertools import count
import time

def remove_to_close_peaks(x_peaks):

    temp = list()

    for i, peak in enumerate(x_peaks):
        if i != 0 and peak - temp[-1] < 8:
            temp.pop(-1)
        temp.append(peak)
    
    return np.asarray(temp) 


def detect_peaks_guassian(df):
    if df is None: return

    df.reset_index(inplace=True)

    dataFiltered = gaussian_filter1d(df.Close, sigma=0.5)
    x_peaks = signal.argrelmax(dataFiltered)[0]

    x = len(x_peaks)
    
    tuned_peaks = tune_peaks(df, x_peaks)
    cleaned_peaks = remove_to_close_peaks(tuned_peaks)

    x1 = len(cleaned_peaks)
    
    if len(cleaned_peaks) > 3:
        return cleaned_peaks
    else: 
        return False


def detect_peaks_cwt(df):
    
    w1 = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    w2 = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    

    peak_list = list()

    for x in w1:
        for x1 in w2:

            try:
                x_peaks = find_peaks_cwt(df.Close, widths=np.arange(x, x1))
            except:
                continue

            x_peaks = remove_to_close_peaks(x_peaks)


            if len(x_peaks) > 3 and len(x_peaks) < 10:
                peak_list.append(x_peaks)

    
    if len(peak_list) > 0: 
        return min(peak_list,key=len)
    else: 
        return False
    
         

def tune_peaks(df, x_peaks):

    if x_peaks is False: return

    # get the highest value previous 10 candles / 10 future candles from peak

    x_peaks_np = list()

    df.reset_index(inplace=True)

    for peak in x_peaks:

        previous = peak - 30
        forward = peak + 30
        #print(f'index before tuning: {peak}')
        highest_price = df.loc[previous:forward, 'Close'].idxmax()
        #print(f'index after tuning: {highest_price}')

        x_peaks_np.append(highest_price)

    return x_peaks_np


        

def all_combi_af_peaks(x_peaks):

    if x_peaks is False: return


    x_peaks_combinations_list = list()

    for n in range(len(x_peaks) +1):

        x_peaks_combinations_list += list(combinations(x_peaks, 3))

    x_peaks_combinations_list.sort(key=lambda tup: tup[1])
    x_peaks_combinations_list = list(dict.fromkeys(x_peaks_combinations_list))
    
    for i, ele in enumerate(x_peaks_combinations_list):
        if ele[0] == ele[1] == ele[2]:
            x_peaks_combinations_list.pop(i)

    assert all(len(tup) == 3 for tup in x_peaks_combinations_list), f"Some Tuples with != 3"

    
    return x_peaks_combinations_list



def fetch_y_values_peaks(df, x_peaks_combinations_list):

    if x_peaks_combinations_list is None: return

    assert all(len(tup) == 3 for tup in x_peaks_combinations_list), f"Some Tuples with != 3"


    y_peaks_combination_list = list()
    
    for y in x_peaks_combinations_list:

        if df.iloc[y[0]].Open <= df.iloc[y[0]].Close:
            x1 = df.iloc[y[0]].Close
        elif df.iloc[y[0]].Open > df.iloc[y[0]].Close: 
            x1 = df.iloc[y[0]].Open

        if df.iloc[y[1]].Open <= df.iloc[y[1]].Close:
            x2 = df.iloc[y[1]].Close
        elif df.iloc[y[1]].Open > df.iloc[y[1]].Close:
            x2 = df.iloc[y[1]].Open

    
        if df.iloc[y[2]].Open <= df.iloc[y[2]].Close:
            x3 = df.iloc[y[2]].Close
        elif df.iloc[y[2]].Open > df.iloc[y[2]].Close:
            x3 = df.iloc[y[2]].Open

        temp = (x1, x2, x3)
        y_peaks_combination_list.append(temp)

    
    assert all(len(tup) == 3 for tup in y_peaks_combination_list), f'Some Tuples with != 3'
    assert len(y_peaks_combination_list) != 0, f'empty list - (fetch_y_values_peaks'

    return y_peaks_combination_list



def peak_regression(x_peak_combinations_list, y_peaks_combination_arr):

    if x_peak_combinations_list is None: return

    trendl_candidates_df= pd.DataFrame(columns =['df_start_index', 'df_end_index' ,'slope', 'intercept', 'r_value', 'p_value', 'std_err'])

    for i in range(len(x_peak_combinations_list)):

        slope, intercept, r_value, p_value, std_err  = linregress(x_peak_combinations_list[i], y_peaks_combination_arr[i])
        
        trendl_candidates_df.loc[i, 'df_start_index'] = x_peak_combinations_list[i][0]
        trendl_candidates_df.loc[i, 'df_end_index'] = x_peak_combinations_list[i][2]
        trendl_candidates_df.loc[i, 'slope'] = slope
        trendl_candidates_df.loc[i, 'intercept'] = intercept
        trendl_candidates_df.loc[i, 'r_value'] = r_value
        trendl_candidates_df.loc[i, 'p_value'] = p_value
        trendl_candidates_df.loc[i, 'std_err'] = std_err

    trendl_candidates_df.sort_values('r_value', inplace=True)


    assert len(trendl_candidates_df) != 0, f'No Trendl candidates - peak_regression'


    return trendl_candidates_df



def fetch_trendl_start_end_price(df, trendl_candidate_df):


    if trendl_candidate_df is None: return
    
    start_price = list()
    start_date = list()

    for row in trendl_candidate_df.df_start_index:
        
        start_price.append(df.iloc[row].Close)
        start_date.append(df.iloc[row].Date)
    
    trendl_candidate_df['start_price'] = start_price
    trendl_candidate_df['start_date'] = pd.to_datetime(start_date, format='%Y-%m-%d %H:%M:%S')


    end_price = list()

    for row in trendl_candidate_df.df_end_index:
        
        end_price.append(df.iloc[row].Close)

    trendl_candidate_df['end_price'] = end_price


    assert len(trendl_candidate_df) != 0, f'No trendl candidates - (fetch_trendl_start_end_price)'

    return trendl_candidate_df



def trendline_angle_degree(trendl_candidates_df):

    if trendl_candidates_df is None: return

    assert len(trendl_candidates_df) != 0, f'No trendl candadates - (trendline_angle_degree)'

    trendl_candidates_df['angle_degree'] = trendl_candidates_df.slope.apply(lambda x: math.degrees(math.atan(x)))
    
    #print(trendl_candidates_df.angle_degree)

    if trendl_candidates_df.empty:
        return None
    else: 
        return trendl_candidates_df



def area_under_trendl():
    pass



def check_trendl_parameters(trendl_candidates_df):

    if trendl_candidates_df is None: return

    trendl_candidates_df.drop(trendl_candidates_df[trendl_candidates_df.angle_degree > -5].index, inplace=True)
    trendl_candidates_df.drop(trendl_candidates_df[trendl_candidates_df.angle_degree > 0].index, inplace=True)
    trendl_candidates_df.drop(trendl_candidates_df[trendl_candidates_df.r_value > -0.999].index, inplace=True)
    

    #print(trendl_candidates_df)
    #print(f'len after check: {len(trendl_candidates_df)}')

    if trendl_candidates_df.empty:
        #print(f'############################################################################################################')
        return None
    else: 
        print(trendl_candidates_df.angle_degree)
        #time.sleep(1)


        return trendl_candidates_df




def plot_all_trendl(df, final_trendline, x_peaks):

    if final_trendline is None: return

    assert len(final_trendline) != 0 , f'No trendl candidates - (plot_all_trendl)'


    # plot peaks:

    df.reset_index(inplace=True)

    y_peaks = list()
    x_peaks_date = list()

    for peak in x_peaks:   
        y_peaks.append(df.iloc[peak].Close)
        x_peaks_date.append(df.iloc[peak].Date)
    
    plt.scatter(x_peaks, y_peaks, c='green')
    

    # plot alle mulige trendlines:

    for row in final_trendline.iterrows():
        slope = row[1].slope
        intercept = row[1].intercept
        y_hat = slope*df.index + intercept

        #plt.plot(df.index, y_hat, color='blue')
    
    #print(f'Areal under Trendline: {np.trapz(y_peaks, x=x_peaks)}') # Areal


    # Plot best line:  
    print(f'r_value: {final_trendline.r_value}')

   
    plt.plot(df.index, y_hat, color='blue')

    plt.plot(df.Close, '-')

    plt.title('Trend Hunter - ETHUSDT - 1D')
    plt.legend()
    plt.grid()
    #plt.show()

    #-----------------------------
    
    df['scatter'] = np.nan
    
    for i, a in enumerate(x_peaks):
        df.loc[a, 'scatter'] = y_peaks[i]

    df.set_index('Date', inplace=True)
    trendl_plot = list(zip(df.index, y_hat))


    path = '//home/traderblakeq/Python/tradingscripts/trendline_results'
    os.chdir(path)

    ap = fplt.make_addplot(df['scatter'],type='scatter', markersize=70, color='blue')
    fig, axlist = fplt.plot(df, figratio=(16,9), type='candle', style='binance', title='Trend Hunter - ETHUSDT - 15M', alines=dict(alines=trendl_plot) , addplot=ap,  ylabel='Price ($)', volume=True, returnfig=True, savefig=f'{str(df.index[0])}.png')
    #fig, axlist = fplt.plot(df, figratio=(16,9), type='candle', style='binance', title='Trend Hunter - ETHUSDT - 15M', alines=dict(alines=trendl_plot) , addplot=ap,  ylabel='Price ($)', volume=True, returnfig=True)

    
    #fplt.show()

    return True


def test_feed(df):
    if len(df) == 100:
        return True
    else: 
        print(len(df))
        return None
