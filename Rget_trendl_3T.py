from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.signal import argrelmax
import mplfinance as fplt
import pandas as pd
import numpy as np
import math
import json
import time 





def detect_peaks_guassian(price, prev_peak_tup, sigma=0.2):
    """
    Detect peaks from DataFrame.Close series using Gaussian filter.
    :param sigma Standard deviation of Gaussian filter.
    """
    # Hvorfor tomt array her ?
    print(f'before pk alg; {prev_peak_tup}')

    #dataFiltered = gaussian_filter1d(price, sigma=sigma)
    x_peaks = argrelmax(price)[0] #order=2

    if len(x_peaks) == len(prev_peak_tup):
        return False
    
    else:
        
        # Når den er blevet sat her ?
        prev_peak_tup = tuple(x_peaks)
        print(f'after =:  {prev_peak_tup}')
        
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


    x_peaks_comb = list()
                
    x_peaks_comb += set(combinations(x_peaks, 3)).difference(last_comb)
    
    if not x_peaks_comb:
        return
    
    else:
        last_comb += x_peaks_comb
        return x_peaks_comb
    


    
def fetch_y_values_peaks(price , x_peak_comb):
    """
    Return max(df.Close, df.Open) at each peak in peak combinations list.
    :params x_peak_combinations
        List of combinations of length 3.
    """
    if x_peak_comb is None: 
        return
    
    # Extract series of peaks.
    X1, X2, X3 = (list(x) for x in zip(*x_peak_comb))

    # Bundle up values as tuples of len 3.
    y_peak_comb = [tuple(y) for y in
            zip(price[X1], price[X2], price[X3])] 


    return y_peak_comb




def peak_regression(price, x_peak_comb, y_peak_comb, df):
    """
    :param x_peak_combinations
        List of peak index combinations (tuples) of len 3
    :param x_peak_combinations
        List of peak value combinations (tuples) of len 3
    """
    if x_peak_comb is None:
        return None, None, None
    

    for i, (x, y) in enumerate(zip(x_peak_comb, y_peak_comb)):

        slope, intercept, r_value, p_value, std_err  = linregress(x, y, alternative='less')

        if r_value > -0.995:
            continue


        peak_tup = tuple(x_peak_comb[i])
        y_hat = slope*np.arange(0, len(price)) + intercept
        aboveArea_p1_p2, belowArea_p1_p2, aboveArea_p2_p3, belowArea_p2_p3 = calc_integrals(price, y_hat, peak_tup)

        if aboveArea_p1_p2 < 10 and aboveArea_p2_p3 < 10  and abs(belowArea_p1_p2) <= 1500 and abs(belowArea_p2_p3) <= 1500 and abs(belowArea_p1_p2) > 100 and abs(belowArea_p2_p3) > 100:

            df.loc[i, 'start_index'] = x_peak_comb[i][0]
            df.loc[i, 'end_index'] = x_peak_comb[i][-1]
            df.loc[i, 'length'] = x_peak_comb[i][-1] - x_peak_comb[i][0]
            df.loc[i, 'slope'] = slope
            df.loc[i, 'intercept'] = intercept
            df.loc[i, 'r_value'] = r_value
            df.loc[i, 'p_value'] = p_value
            df.loc[i, 'std_err'] = std_err
            df.loc[i, 'angle'] = math.degrees(math.atan(slope))
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

    
    slice_tup = ((0,1), (1,-1))

    res_list = list()

    for tup in slice_tup:

        y1 =price[peak_tup[tup[0]]:peak_tup[tup[1]]+1]
        y2 =y_hat[peak_tup[tup[0]]:peak_tup[tup[1]]+1]
        x = np.arange(0, len(y1))

        dy = y1 - y2
        b0 = dy[:-1]
        b1 = dy[1:]
        b = np.c_[b0, b1]
        r = np.abs(b0) / (np.abs(b0) + np.abs(b1))
        rr = np.c_[r, 1-r]
        dx = np.diff(x)
        h = rr * dx[:, None]
        br = (b * rr[:, ::-1]).sum(1)
        a = (b + br[:, None]) * h / 2
        result = np.sum(a[a > 0]), np.sum(a[a < 0])

        if details is not None:
            details.update(locals())  # for debugging

        res_list.append(result[0])
        res_list.append(result[1])


    return res_list[0], res_list[1], res_list[2], res_list[3]   




def extract_data_for_plotting(close, index, final_trendline, x_peaks, peak_tup, length_list, y_peak_list):
    if final_trendline is None: return

    assert len(final_trendline) != 0 , f'No trendl candidates - (extract_data_for_plotting)'

    
    # Save y peaks to list:
    y_peaks = list()
   
    for peak in x_peaks:
        y_peaks.append(close[peak])


    length_list.append(int(final_trendline.length))
    y_peak_list.append(max(y_peaks))
    #print(f'max length: {max(length_list)}')
    #print(f'min y price: {min(y_peak_list)}')
    #print(f'max y price:{max(y_peak_list)}')



    scatter = np.full(len(index), fill_value=np.nan)
    scatter_act_peak = np.full(len(index), fill_value=np.nan)

    for peak in peak_tup:
        scatter_act_peak[peak] = close[peak]


    for peak in x_peaks:
        scatter[peak] = close[peak]


    return (scatter, scatter_act_peak) 



def plot_final_peaks_and_final_trendline(df, tup_data, y_hat, timestamp, peak_tup, candidates_df, fit_plot=0):

    if tup_data is None: return
    

    scatter, actual_peaks = tup_data[0], tup_data[1]

    df_slice = df[peak_tup[0] : peak_tup[-1]+1]
    y_hat_slice = y_hat[peak_tup[0] : peak_tup[-1]+1]
    scatter_slice = scatter[peak_tup[0] : peak_tup[-1]+1]
    actual_peaks_slice = actual_peaks[peak_tup[0] : peak_tup[-1]+1]


    if fit_plot !=0:
        # fit all plots to same x-axis lenght, for integral area's to be interpret equal visually.

        ext_len = fit_plot - int(candidates_df.length)

        df_slice =  df[ peak_tup[0] : peak_tup[-1] + ext_len + 1].copy()
        y_hat_slice = np.pad(y_hat_slice, (0,ext_len), 'constant', constant_values=np.nan)
        scatter_slice = np.pad(scatter_slice, (0,ext_len), 'constant', constant_values=np.nan)
        actual_peaks_slice = np.pad(actual_peaks_slice, (0,ext_len), 'constant', constant_values=np.nan) 
    

    trendl_plot = list(zip(df_slice.index, y_hat_slice))
   

    path = './trendline_results'

    subplt = list()
    subplt.append(fplt.make_addplot(scatter_slice, type='scatter', markersize=30, color='red'))
    subplt.append(fplt.make_addplot(actual_peaks_slice, type='scatter', markersize=60, color='green'))


    fig, axlist = fplt.plot(df_slice, figratio=(16,9), type='candle', style='binance', title='Trend Hunter - ETHUSDT - 15M', alines=dict(alines=trendl_plot) , addplot=subplt, ylabel='Price ($)', returnfig=True, savefig=f'{path}/{str(timestamp)}.png')
    #fig, axlist = fplt.plot(df_slice, figratio=(16,9), type='candle', style='binance', title='Trend Hunter - ETHUSDT - 15M', ylim=(1100.0, 1650.0), alines=dict(alines=trendl_plot) , addplot=subplt, ylabel='Price ($)', returnfig=True, savefig=f'{path}/{str(timestamp)}.png')

    
    #trendl_details = f'length: {candidates_df.length}, angle: {candidates_df.angle}, area above t1t2: {round(candidates_df.aboveArea_p1_p2,1)}, area below t1t2: {round(candidates_df.belowArea_p1_p2, 1)}, area above t2t3: {round(candidates_df.aboveArea_p2_p3,1)}, Area below t2t3: {round(candidates_df.belowArea_p2_p3,1)}'

    #write_txt = open(f'{path}/{str(timestamp)}.txt',"w")
    #write_txt.write(trendl_details)


    #df.reset_index(inplace=True)

    #plt.scatter(df_slice.index, scatter_slice, c='green')
    #plt.plot(df_slice.index, y_hat_slice, color='blue')
    #plt.plot(df_slice.Close, '-')
    #plt.title('Trend Hunter - ETHUSDT - 1D')
    #plt.grid()

    #fplt.show()


    to_json(df_slice, y_hat)

    return True, df.index[peak_tup[-1]]



def to_json(date, y_hat):

    trendl_dict = dict({
            
            "x1": f"{date.index[0].value//10**9}", 
            "y1": f"{y_hat[0]}",
            "x2": f"{date.index[-1].value//10**9}", 
            "y2": f"{y_hat[-1]}"
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