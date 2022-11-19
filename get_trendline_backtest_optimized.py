import math
import json
import os
import time
from itertools import combinations, count

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import mplfinance as fplt

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks, find_peaks_cwt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress



def remove_to_close_peaks(x_peaks, too_close=6):
    """
    Return new array with peaks that are closer than :param too_close to
    each other removed
    """
    temp = list()

    for i, peak in enumerate(x_peaks):
        if i==0: temp.append(peak)
        elif abs(temp[-1]-peak) < too_close:
            continue
        else: 
            temp.append(peak)
    
    return temp 


def detect_peaks_guassian(df, sigma=2):
    """
    Detect peaks from DataFrame.Close series using Gaussian filter.

    :param sigma
        Standard deviation of Gaussian filter.
    """
    if df is None: 
        return

    peak_list = list()

    dataFiltered = gaussian_filter1d(df.Close, sigma=sigma)
    x_peaks = signal.argrelmax(dataFiltered)[0]
    tuned_peaks = tune_peaks(df, x_peaks)
    cleaned_peaks = remove_to_close_peaks(x_peaks)

    peak_list.append(cleaned_peaks)

    if len(peak_list) > 0:
        return min(peak_list,key=len)

    else: 
        return False




def tune_peaks(df, x_peaks):
    """
    Get the highest value of previous x candles / x future candles 
    from peak
    """
    if x_peaks is False: 
        return

    x_peaks_np = list()

    df.reset_index(inplace=True)

    for peak in x_peaks:
        previous = peak - 20
        forward = peak + 20

        highest_price = df.loc[previous:forward, 'Close'].idxmax()

        x_peaks_np.append(highest_price)

    return x_peaks_np


        

def all_combi_af_peaks(x_peaks):
    """
    Return list of all distinct combinations of length of 3 of :param 
    x_peaks.
    """
    if x_peaks is False: 
        return

    x_peaks_combinations_list = list()

    x_peaks_combinations_list += list(combinations(x_peaks, 3))

    x_peaks_combinations_list.sort(key=lambda tup: tup[1])
    x_peaks_combinations_list = list(dict.fromkeys(
            x_peaks_combinations_list))
    
    # Remove any non distinct combinations.
    ([x_peaks_combinations_list.pop(a) for a, i in enumerate(x_peaks_combinations_list) if i[0]==i[1]==i[2]])

    if len(x_peaks_combinations_list) == 0: 
        return None
    else:
        return x_peaks_combinations_list



def fetch_y_values_peaks(df, x_peak_combinations):
    """
    Return max(df.Close, df.Open) at each peak in peak combinations list.

    :params x_peak_combinations
        List of combinations of length 3.
    """
    if x_peak_combinations is None: 
        return

    #assert all(len(tup) == 3 for tup in x_peak_combinations), \
    #        f"Some Tuples with len != 3"


    # Assign Y value to the highest of Open and Close at all peaks.
    # The resulting series, s_max is a numpy array.
    s_max = np.maximum(df.Open, df.Close)

    # Extract series of peaks.
    X = zip(*x_peak_combinations)
    X1, X2, X3 = (list(x) for x in X)

    # Bundle up values as tuples of len 3.
    y_peak_combinations = [tuple(y) for y in
            zip(s_max[X1], s_max[X2], s_max[X3])] 


    return y_peak_combinations



def peak_regression(df, x_peak_combinations, y_peak_combinations):
    """
    :param x_peak_combinations
        List of peak index combinations (tuples) of len 3
    :param x_peak_combinations
        List of peak value combinations (tuples) of len 3
    """
    
    if x_peak_combinations is None: 
        return pd.DataFrame()


    for i in range(len(x_peak_combinations)):
        slope, intercept, r_value, p_value, std_err  = linregress(
                x_peak_combinations[i], y_peak_combinations[i])
        
        if r_value < -0.999:
            df.loc[i, 'df_start_index'] = x_peak_combinations[i][0]
            df.loc[i, 'df_end_index'] = x_peak_combinations[i][2]
            df.loc[i, 'slope'] = slope
            df.loc[i, 'intercept'] = intercept
            df.loc[i, 'r_value'] = r_value
            df.loc[i, 'p_value'] = p_value
            df.loc[i, 'std_err'] = std_err

            df.sort_values('r_value', inplace=True)

            #assert len(trendl_candidates_df) != 0, \
            #   f'No Trendl candidates - peak_regression'

    
    return df



def fetch_trendl_start_end_price(df, trendl_candidate_df):

    if trendl_candidate_df is None: 
        return

    df.reset_index(inplace=True)

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


    assert len(trendl_candidate_df) != 0, \
            f'No trendl candidates - (fetch_trendl_start_end_price)'

    return trendl_candidate_df



def trendline_angle_degree(trendl_candidates_df):
    if trendl_candidates_df is None: 
        return

    assert len(trendl_candidates_df) != 0, \
            f'No trendl candadates - (trendline_angle_degree)'

    trendl_candidates_df['angle_degree'] = trendl_candidates_df \
            .slope.apply(lambda x: math.degrees(math.atan(x)))
    
    if trendl_candidates_df.empty:
        return None
    else: 
        return trendl_candidates_df



def area_under_trendl():
    pass



def check_trendl_parameters(trendl_candidates_df):

    if trendl_candidates_df is None: return

    trendl_candidates_df.drop(trendl_candidates_df[
            trendl_candidates_df.angle_degree > -5].index, inplace=True)
    trendl_candidates_df.drop(trendl_candidates_df[
            trendl_candidates_df.angle_degree > 0].index, inplace=True)
    trendl_candidates_df.drop(trendl_candidates_df[
        trendl_candidates_df.r_value > -0.999].index, inplace=True)
    
    if trendl_candidates_df.empty:
        return None

    else: 
        return trendl_candidates_df.iloc[0]



def extract_data_for_plotting(df, final_trendline, x_peaks):
    if final_trendline is None: return

    assert len(final_trendline) != 0 , f'No trendl candidates - (extract_data_for_plotting)'

    df.reset_index(inplace=True)

    # Save x,y peaks to list:

    y_peaks_date = list()
    y_peaks = list()

    for peak in x_peaks:   
        y_peaks_date.append(df.iloc[peak].Date)
        y_peaks.append(df.iloc[peak].Close)

    # Calcualte x, y trendline slope:

    slope = final_trendline.slope
    intercept = final_trendline.intercept
    y_hat = slope*df.index + intercept

    # Fill scatter row for plotting:

    df['scatter'] = np.nan

    for i, a in enumerate(x_peaks):

        df.iloc[a, -1] = y_peaks[i]

    return (df, y_peaks_date, y_peaks, y_hat) 



def plot_final_peaks_and_final_trendline(df, tup_data, x_peaks):

    if tup_data is None: return

    df_scatter, y_peaks_date, y_peaks, y_hat = tup_data[0], tup_data[1], tup_data[2], tup_data[3]
    
    trendl_plot = list(zip(df.Date, y_hat))
   
    trendl_start_end = list([trendl_plot[0], trendl_plot[-1]])
    
    trendl_dict = dict({
        
            "x1": f"{trendl_start_end[0][0].value//10**9}", 
            "y1": f"{trendl_start_end[0][1]}",
            "x2": f"{trendl_start_end[-1][0].value//10**9}", 
            "y2": f"{trendl_start_end[-1][1]}"
        })

    with open("data.json", "r") as f:
        data_list = json.load(f)
    data_list.append(trendl_dict)

    with open('data.json', 'w') as outfile:
        json.dump(data_list, outfile)
    
    path = './trendline_results'
    df.set_index(df.Date, inplace=True)
    ap = fplt.make_addplot(df_scatter['scatter'],type='scatter', markersize=70, color='blue')
    fig, axlist = fplt.plot(df, figratio=(16,9), type='candle', style='binance', title='Trend Hunter - ETHUSDT - 15M', alines=dict(alines=trendl_plot) , addplot=ap,  ylabel='Price ($)', returnfig=True, savefig=f'{path}/{str(df.index[0])}.png')
    #fig, axlist = fplt.plot(df, figratio=(16,9), type='candle', style='binance', title='Trend Hunter - ETHUSDT - 15M', alines=dict(alines=trendl_plot) , addplot=ap,  ylabel='Price ($)', returnfig=True)

    plt.scatter(x_peaks, y_peaks, c='green')
    plt.plot(df.index, y_hat, color='blue')
    plt.plot(df.Close, '-')
    plt.title('Trend Hunter - ETHUSDT - 1D')
    plt.grid()

    #fplt.show()


    return True



def test_feed(df):
    if len(df) == 1000:
        return True

    else: 
        print(len(df))
        print(df.index)
        return None
