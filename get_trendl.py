from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
from itertools import combinations
import matplotlib.pyplot as plt
from operator import itemgetter
from pandas import DataFrame
import mplfinance as fplt
from scipy import signal
import pandas as pd
import numpy as np
import math
import json
import time





def detect_peaks_guassian(price, sigma=0.5):
    """
    Detect peaks from DataFrame.Close series using Gaussian filter.
    :param sigma Standard deviation of Gaussian filter.
    """

    if price is None: 
        return

    #dataFiltered = gaussian_filter1d(price, sigma=sigma)
    x_peaks = signal.argrelmax(price, order=2)[0]

    if len(x_peaks) >= 3:
        return x_peaks

    else: 
        return False




def all_combi_af_peaks(x_peaks, last_comb):
    """
    Return list of all distinct combinations of length of 3 of :param 
    x_peaks.
    """

    if x_peaks is False: 
        return


    x_peaks_combination = list()
                
    x_peaks_combination += set(list(combinations(x_peaks, 3))).difference(last_comb)

    last_comb += x_peaks_combination


    if len(x_peaks_combination)>0:
        return x_peaks_combination
    else: 
        return



def fetch_y_values_peaks(price , x_peak_combinations):
    """
    Return max(df.Close, df.Open) at each peak in peak combinations list.
    :params x_peak_combinations
        List of combinations of length 3.
    """
    if x_peak_combinations is None: 
        return
    
    # Extract series of peaks.
    X = zip(*x_peak_combinations)
    X1, X2, X3 = (list(x) for x in X)

    # Bundle up values as tuples of len 3.
    y_peak_combinations = [tuple(y) for y in
            zip(price[X1], price[X2], price[X3])] 


    return y_peak_combinations



def peak_regression(price, x_peak_combinations, y_peak_combinations):
    """
    :param x_peak_combinations
        List of peak index combinations (tuples) of len 3
    :param x_peak_combinations
        List of peak value combinations (tuples) of len 3
    """
    if x_peak_combinations is None:
        return None, None, None
    

    df = DataFrame()


    for i in range(len(x_peak_combinations)):
        slope, intercept, r_value, p_value, std_err  = linregress(
                x_peak_combinations[i], y_peak_combinations[i], alternative='less')

        angle = math.degrees(math.atan(slope))
        
        if r_value < -0.995:
          
            peak_tup = tuple(x_peak_combinations[i])
            y_hat = slope*np.arange(0, len(price)) + intercept
            aboveArea_p1_p2, belowArea_p1_p2, aboveArea_p2_p3, belowArea_p2_p3 = calc_integrals(price, y_hat, peak_tup)

            if aboveArea_p1_p2 < 10 and aboveArea_p2_p3 < 10  and abs(belowArea_p1_p2) <= 1500 and abs(belowArea_p2_p3) <= 1500 and abs(belowArea_p1_p2) > 300 and abs(belowArea_p2_p3) > 300:

                df.loc[i, 'start_index'] = x_peak_combinations[i][0]
                df.loc[i, 'end_index'] = x_peak_combinations[i][2]
                df.loc[i, 'length'] = x_peak_combinations[i][2] - x_peak_combinations[i][0]
                df.loc[i, 'slope'] = slope
                df.loc[i, 'intercept'] = intercept
                df.loc[i, 'r_value'] = r_value
                df.loc[i, 'p_value'] = p_value
                df.loc[i, 'std_err'] = std_err
                df.loc[i, 'angle'] = angle
                df.loc[i, 'aboveArea_p1_p2'] = aboveArea_p1_p2
                df.loc[i, 'belowArea_p1_p2'] = belowArea_p1_p2
                df.loc[i, 'aboveArea_p2_p3'] = aboveArea_p2_p3
                df.loc[i, 'belowArea_p2_p3'] = belowArea_p2_p3
              

                return df, peak_tup, y_hat
    
    if df.empty:
        return None, None, None
    else:
        return df, peak_tup, y_hat


def calc_integrals(price, y_hat, peak_tup, details=None):

    y1 =price[peak_tup[0]:peak_tup[1]+1]
    y2 =y_hat[peak_tup[0]:peak_tup[1]+1]
    x1 = np.arange(0, len(y1))

    y3 =price[peak_tup[1]:peak_tup[2]+1]
    y4 =y_hat[peak_tup[1]:peak_tup[2]+1]
    x2 = np.arange(0, len(y3))

    y_list = [(x1, y1, y2), (x2, y3, y4)]

    res_list = list()

    for integ in y_list:

        dy = integ[1] - integ[2]
        b0 = dy[:-1]
        b1 = dy[1:]
        b = np.c_[b0, b1]
        r = np.abs(b0) / (np.abs(b0) + np.abs(b1))
        rr = np.c_[r, 1-r]
        dx = np.diff(integ[0])
        h = rr * dx[:, None]
        br = (b * rr[:, ::-1]).sum(1)
        a = (b + br[:, None]) * h / 2
        result = np.sum(a[a > 0]), np.sum(a[a < 0])

        if details is not None:
            details.update(locals())  # for debugging

        res_list.append(result[0])
        res_list.append(result[1])


    return res_list[0], res_list[1], res_list[2], res_list[3]




def check_trendl_parameters(candidates_df):

    if candidates_df is None: return

    #candidates_df.drop(candidates_df[
    #        candidates_df.angle_degree > -5].index, inplace=True)
    
    candidates_df.drop(candidates_df[
            candidates_df.angle_degree > 0].index, inplace=True)
    
    #candidates_df.drop(candidates_df[
    #    candidates_df.r_value > -0.999].index, inplace=True)

    #candidates_df.drop(candidates_df[
    #    candidates_df.pos_area > 10].index, inplace=True)

    #candidates_df.drop(candidates_df[
    #    candidates_df.neg_area > 500].index, inplace=True)


    candidates_df.sort_values('r_value', inplace=True)


    if candidates_df.empty:
        return None
    else: 
        
        #print(f'Pos Area 1&2: {candidates_df.pos_area_p1_p2.iloc[0]}')
        #print(f'Neg Area 1&2: {candidates_df.neg_area_p1_p2.iloc[0]}')

        #print(f'Pos Area 2&3: {candidates_df.pos_area_p2_p3.iloc[0]}')
        #print(f'Neg Area 2&3: {candidates_df.neg_area_p2_p3.iloc[0]}')

        return candidates_df.iloc[0]



def extract_data_for_plotting(close, index, final_trendline, x_peaks, peak_tup, length_list, y_peak_list):
    if final_trendline is None: return

    assert len(final_trendline) != 0 , f'No trendl candidates - (extract_data_for_plotting)'

    
    # Save x,y peaks to list:
    y_peaks_date = list()
    y_peaks = list()
   
    for peak in x_peaks:
        y_peaks_date.append(index[peak])
        y_peaks.append(close[peak])


    length_list.append(int(final_trendline.length))
    y_peak_list.append(max(y_peaks))
    print(f'max length: {max(length_list)}')
    print(f'min y price: {min(y_peak_list)}')
    print(f'max y price:{max(y_peak_list)}')



    scatter = np.full(len(index), fill_value=np.nan)
    scatter_act_peak = np.full(len(index), fill_value=np.nan)

    for peak in peak_tup:
        scatter_act_peak[peak] = close[peak]
    

    for peak in x_peaks:
        scatter[peak] = close[peak]


    return (scatter, y_peaks_date, y_peaks, scatter_act_peak) 



def plot_final_peaks_and_final_trendline(df, tup_data, y_hat, timestamp, peak_tup, candidates_df, fit_plot=0):

    if tup_data is None: return
    

    scatter, y_peaks_date, y_peaks, actual_peaks = tup_data[0], tup_data[1], tup_data[2], tup_data[3]

    df_slice = df[peak_tup[0] : peak_tup[2]+1]
    y_hat_slice = y_hat[peak_tup[0] : peak_tup[2]+1]
    scatter_slice = scatter[peak_tup[0] : peak_tup[2]+1]
    actual_peaks_slice = actual_peaks[peak_tup[0] : peak_tup[2]+1]
    

    if fit_plot !=0:

        ext_len = fit_plot - int(candidates_df.length)

        df_slice = df[ peak_tup[0] : peak_tup[2] + ext_len + 1].copy()

        y_hat_slice = np.pad(y_hat_slice, (0,ext_len), 'constant', constant_values=np.nan)
        scatter_slice = np.pad(scatter_slice, (0,ext_len), 'constant', constant_values=np.nan)
        actual_peaks_slice = np.pad(actual_peaks_slice, (0,ext_len), 'constant', constant_values=np.nan) 

    
    trendl_plot = list(zip(df_slice.index, y_hat_slice))
   
    trendl_start_end = list([trendl_plot[0], trendl_plot[-1]])

    path = './trendline_results'
    
    subplt = list()
    subplt.append(fplt.make_addplot(scatter_slice, type='scatter', markersize=70, color='red'))
    subplt.append(fplt.make_addplot(actual_peaks_slice, type='scatter', markersize=70, color='green'))

    #plot_string = f'length: {str(candidates_df.length)}, angle: {str(candidates_df.angle)}, area above t1t2: {str(round(candidates_df.aboveArea_p1_p2,1))}, area below t1t2: {str(round(candidates_df.belowArea_p1_p2, 1))}, area above t2t3: {str(round(candidates_df.aboveArea_p2_p3,1))}, Area below t2t3: {str(round(candidates_df.belowArea_p2_p3,1))} '

    fig, axlist = fplt.plot(df_slice, figratio=(16,9), type='candle', style='binance', title='Trend Hunter - ETHUSDT - 15M', alines=dict(alines=trendl_plot) , addplot=subplt, ylabel='Price ($)', returnfig=True, savefig=f'{path}/{str(timestamp)}.png')
    #fig, axlist = fplt.plot(df_slice, figratio=(16,9), type='candle', style='binance', title='Trend Hunter - ETHUSDT - 15M', ylim=(1100.0, 1650.0), alines=dict(alines=trendl_plot) , addplot=subplt, ylabel='Price ($)', returnfig=True, savefig=f'{path}/{str(timestamp)}.png')

    #df.reset_index(inplace=True)

    #plt.scatter(x_peaks, y_peaks, c='green')
    #plt.plot(df.index, y_hat, color='blue')
    #plt.plot(df.Close, '-')
    #plt.title('Trend Hunter - ETHUSDT - 1D')
    #plt.grid()

    #fplt.show()

    to_json(trendl_start_end)

    return True



def to_json(tup):


    trendl_dict = dict({
        
            "x1": f"{tup[0][0].value//10**9}", 
            "y1": f"{tup[0][1]}",
            "x2": f"{tup[-1][0].value//10**9}", 
            "y2": f"{tup[-1][1]}"
        })

    with open("trendlines_testdata.json", "r") as f:
        data_list = json.load(f)
    data_list.append(trendl_dict)

    with open('trendlines_testdata.json', 'w') as outfile:
        json.dump(data_list, outfile)


    return True



def test_feed(data):
    if len(data) == 1000:
        return True

    else: 
        print(len(data))
        return None