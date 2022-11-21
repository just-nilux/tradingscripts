from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
from itertools import combinations
import matplotlib.pyplot as plt
from pandas import DataFrame
import mplfinance as fplt
from scipy import signal
import pandas as pd
import numpy as np
import math
import json
import time






def detect_peaks_guassian(price, sigma=2):
    """
    Detect peaks from DataFrame.Close series using Gaussian filter.
    :param sigma Standard deviation of Gaussian filter.
    """

    if price is None: 
        return

    dataFiltered = gaussian_filter1d(price, sigma=sigma)
    x_peaks = signal.argrelmax(dataFiltered)[0]

    if len(x_peaks) < 3:
        return False

    tuned_peaks = tune_peaks(price, x_peaks)
    cleaned_peaks = remove_too_close_peaks(tuned_peaks)


    if len(cleaned_peaks) >= 3:
        return cleaned_peaks

    else: 
        return False



def tune_peaks(close, x_peaks):
    """
    Get the highest value of previous 30 candles / 30 future candles 
    from peak
    """

    if x_peaks is False: 
        return


    x_peaks_ = list()
    indices  = np.arange(len(close))
    close = np.array(close)

    for peak in x_peaks:

        previous = peak - 10
        forward = peak + 10

        if previous <= 0:
            sliced = close[0:forward]            
            sliced_idx = indices[0:forward]
          
            argmax = sliced_idx[sliced.argmax()]

        elif forward > len(close):
            sliced = close[previous:]
            sliced_idx = indices[previous:]
           
            argmax = sliced_idx[sliced.argmax()]

        else:
            sliced = close[previous:forward]
            sliced_idx = indices[previous:forward]
   
            argmax = sliced_idx[sliced.argmax()]
        
    
        x_peaks_.append(argmax)


    return x_peaks_



def remove_too_close_peaks(x_peaks, too_close=6):

    temp = list()

    for i, peak in enumerate(x_peaks):
        if i==0: temp.append(peak)
        elif abs(temp[-1]-peak) < too_close:
            continue
        else: 
            temp.append(peak)
    
    return temp



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
    
    # Remove any non distinct combinations
    ([x_peaks_combinations_list.pop(a) for a, i in enumerate(x_peaks_combinations_list) if i[0]==i[1]==i[2]])

    #assert all(len(tup) == 3 for tup in x_peaks_combinations_list), \
    #        f"Some Tuples with != 3"

       #------------------------------------

    return x_peaks_combinations_list



def fetch_y_values_peaks(price , x_peak_combinations):
    """
    Return max(df.Close, df.Open) at each peak in peak combinations list.
    :params x_peak_combinations
        List of combinations of length 3.
    """
    if x_peak_combinations is None: 
        return

    #assert all(len(tup) == 3 for tup in x_peak_combinations), \
    #        f"Some Tuples with len != 3"
    
    # Extract series of peaks.
    X = zip(*x_peak_combinations)
    X1, X2, X3 = (list(x) for x in X)
    
    # Bundle up values as tuples of len 3.
    y_peak_combinations = [tuple(y) for y in
            zip(price[X1], price[X2], price[X3])] 


    return y_peak_combinations



def peak_regression(x_peak_combinations, y_peak_combinations):
    """
    :param x_peak_combinations
        List of peak index combinations (tuples) of len 3
    :param x_peak_combinations
        List of peak value combinations (tuples) of len 3
    """
    df = DataFrame(columns =['start_index', 'end_index' ,'slope', 'intercept', 'r_value', 'p_value', 'std_err'])

    if x_peak_combinations is None: 
        return pd.DataFrame()

    for i in range(len(x_peak_combinations)):
        slope, intercept, r_value, p_value, std_err  = linregress(
                x_peak_combinations[i], y_peak_combinations[i])
        
        if r_value < -0.999:
            peak_tup = tuple(x_peak_combinations[i])

            df.loc[i, 'start_index'] = x_peak_combinations[i][0]
            df.loc[i, 'end_index'] = x_peak_combinations[i][2]
            df.loc[i, 'slope'] = slope
            df.loc[i, 'intercept'] = intercept
            df.loc[i, 'r_value'] = r_value
            df.loc[i, 'p_value'] = p_value
            df.loc[i, 'std_err'] = std_err

            #assert len(trendl_candidates_df) != 0, \
            #   f'No Trendl candidates - peak_regression'
    
    if df.empty:
        return None, None
    else:
        return df, peak_tup



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

    trendl_candidates_df.sort_values('r_value', inplace=True)
    
    if trendl_candidates_df.empty:
        return None

    else: 
        return trendl_candidates_df.iloc[0]


def extract_data_for_plotting_numpy(close, index, final_trendline, x_peaks):
    if final_trendline is None: return

    assert len(final_trendline) != 0 , f'No trendl candidates - (extract_data_for_plotting)'

    # Save x,y peaks to list:

    y_peaks_date = list()
    y_peaks = list()
   
    for peak in x_peaks:
        y_peaks_date.append(index[peak])
        y_peaks.append(close[peak])

    # Calcualte x, y trendline slope:
    slope = final_trendline.slope
    intercept = final_trendline.intercept
    y_hat = slope*np.arange(0, len(index)) + intercept

    scatter = list()
    # der er tit mere end 3 fundne toppe!.
    for i, price in enumerate(close):

        if y_peaks[0] == price:
            scatter.append(price)
        elif y_peaks[1] == price:
            scatter.append(price)
        elif y_peaks[2] == price:
            scatter.append(price)
        else:
            scatter.append(np.nan)

    return (scatter, y_peaks_date, y_peaks, y_hat) 



def plot_final_peaks_and_final_trendline(df, tup_data, x_peaks):

    if tup_data is None: return
    
    scatter, y_peaks_date, y_peaks, y_hat = tup_data[0], tup_data[1], tup_data[2], tup_data[3]
    
    trendl_plot = list(zip(df.index, y_hat))
   
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
    
    ap = fplt.make_addplot(scatter,type='scatter', markersize=70, color='blue')
    fig, axlist = fplt.plot(df, figratio=(16,9), type='candle', style='binance', title='Trend Hunter - ETHUSDT - 15M', alines=dict(alines=trendl_plot) , addplot=ap,  ylabel='Price ($)', returnfig=True, savefig=f'{path}/{str(y_peaks_date[0])}.png')
    #fig, axlist = fplt.plot(df, figratio=(16,9), type='candle', style='binance', title='Trend Hunter - ETHUSDT - 15M', alines=dict(alines=trendl_plot) , addplot=ap,  ylabel='Price ($)', returnfig=True)
    
    df.reset_index(inplace=True)

    plt.scatter(x_peaks, y_peaks, c='green')
    plt.plot(df.index, y_hat, color='blue')
    plt.plot(df.Close, '-')
    plt.title('Trend Hunter - ETHUSDT - 1D')
    plt.grid()

    #fplt.show()


    return True



def test_feed(data):
    if len(data) == 1000:
        return True

    else: 
        print(len(data))
        return None