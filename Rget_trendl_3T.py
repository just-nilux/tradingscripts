from numpy import sum as npsum
from itertools import combinations
from numpy import corrcoef, isclose
import matplotlib.pyplot as plt
import mplfinance as fplt
from numba import jit
from numba import typed

import numpy as np
from math import atan, degrees
import json
import time 


@jit
def mul_xor_hash(arr, init=65537, k=37):
    result = init
    for x in arr.view(np.uint64):
        result = (result * k) ^ x
    return result


@jit
def setdiff2d_nb(arr1, arr2):
    # : build `delta` set using hashes
    delta = {mul_xor_hash(arr2[0])}
    for i in range(1, arr2.shape[0]):
        delta.add(mul_xor_hash(arr2[i]))
    # : compute the size of the result
    n = 0
    for i in range(arr1.shape[0]):
        if mul_xor_hash(arr1[i]) not in delta:
            n += 1
    # : build the result
    result = np.empty((n, arr1.shape[-1]), dtype=arr1.dtype)
    j = 0
    for i in range(arr1.shape[0]):
        if mul_xor_hash(arr1[i]) not in delta:
            result[j] = arr1[i]
            j += 1
    return result




def all_combi_af_peaks(x_peaks, last_comb):
    """
    Return list of all distinct combinations of length of 3 of :param 
    x_peaks.

    last_comb = list()
    """

    x_peaks_comb = list(set(combinations(x_peaks, 3)).difference(last_comb))
    
    if not x_peaks_comb:
        return None
    
    else:
        last_comb += x_peaks_comb
        return x_peaks_comb
    

@jit('(int64, int64)')
def combCount(n, r):
    if r < 0:
        return 0
    res = 1
    if r > n - r:
        r = n - r
    for i in range(r):
        res *= (n - i)
        res //= (i + 1)
    return res


@jit(inline='always')
def genComb_x3(arr):
    n = arr.size
    nComb = combCount(n, 3)
    out = np.empty((nComb, 3), dtype=arr.dtype)
    a, b, c = 0, 1, 2
    arr_a = arr[a]
    arr_b = arr[b]
    for cur in range(nComb):
        out[cur, 0] = arr_a
        out[cur, 1] = arr_b
        out[cur, 2] = arr[c]
        if c < n - 1:
            c += 1
        else:
            if b < n - 2:
                b, c = b + 1, b + 2
                arr_b = arr[b]
            else:
                a, b, c = a + 1, a + 2, a + 3
                arr_a = arr[a]
                arr_b = arr[b]
    
    return out


def fetch_y_values_peaks(price , x_peak_comb):

    """
    Return price at each peak in x_peak_comb array.
    :params x_peak_combinations
        array of combinations of length 3.
    """
    if x_peak_comb is None: 
        return None
    

    # Extract series of peaks.
    #X1, X2, X3 = [np.array(x) for x in zip(*x_peak_comb)]

    # Extract series of peaks using NumPy advanced indexing.
    X1, X2, X3 = np.array(x_peak_comb).T

    # Bundle up values as arrays of len 3.
    #y_peak_comb = [np.array(y) for y in
    #    zip(price[X1], price[X2], price[X3])] 
    
    y_peak_comb = np.column_stack((price[X1], price[X2], price[X3]))

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
    

    for i, (x, y) in enumerate(zip(np.array(x_peak_comb), y_peak_comb)):
        

        try:
            r_value, slope, intercept = vectorized_linregress(x,y)
        except ZeroDivisionError as e:
            e
        
        if r_value is None:
            continue

        peak_tup = tuple(x_peak_comb[i])
        y_hat = slope*np.arange(0, len(price)) + intercept
        aboveArea_p1_p2, belowArea_p1_p2, aboveArea_p2_p3, belowArea_p2_p3 = calc_integrals(price, y_hat, peak_tup)

        if aboveArea_p1_p2 < 10 and aboveArea_p2_p3 < 10  and abs(belowArea_p1_p2) < 1000 and abs(belowArea_p2_p3) < 1000 and abs(belowArea_p1_p2) > 100 and abs(belowArea_p2_p3) > 100:

            df.loc[i, 'start_index'] = x_peak_comb[i][0]
            df.loc[i, 'end_index'] = x_peak_comb[i][-1]
            df.loc[i, 'length'] = x_peak_comb[i][-1] - x_peak_comb[i][0]
            df.loc[i, 'slope'] = slope
            df.loc[i, 'intercept'] = intercept
            df.loc[i, 'r_value'] = abs(r_value)
            df.loc[i, 'angle'] = degrees(atan(slope))
            df.loc[i, 'aboveArea_p1_p2'] = aboveArea_p1_p2
            df.loc[i, 'belowArea_p1_p2'] = belowArea_p1_p2
            df.loc[i, 'aboveArea_p2_p3'] = aboveArea_p2_p3
            df.loc[i, 'belowArea_p2_p3'] = belowArea_p2_p3
            #print(f'r_val: {abs(r_value)}')
            #print(f'belowA - p1: {abs(belowArea_p1_p2)}')
            #print(f'belowA - p2: {abs(belowArea_p2_p3)}')
                
            return df, peak_tup, y_hat
    
    if df.empty:
        return None, None, None
    else:
        return df, peak_tup, y_hat


from numpy import corrcoef, isclose

@jit(inline='always')
def vectorized_linregress_chat(x, y):
    
    r_val = corrcoef(x, y)[0][1]
    if r_val>.995:
        return None, None, None


    n = x.shape[0]
    #1. Compute data length, mean and standard deviation along time axis for further use: 
    xmean = x.mean()
    ymean = y.mean()
    xstd  = x.std()
    
    #2. Compute covariance along time axis
    cov   =  npsum((x - xmean)*(y - ymean))/(n)

    #4. Compute regression slope and intercept:
    slope     = cov/(xstd**2)
    intercept = ymean - xmean*slope  
    
    return r_val, slope, intercept


@jit(inline='always')
def vectorized_linregress(x, y):
    """Linear regression calc (Vectorized)"""
    
    #1. Compute data length, mean and standard deviation along time axis for further use: 
    n     = x.shape[0]
    xmean = x.mean()
    ymean = y.mean()
    xstd  = x.std()
    ystd  = y.std()
    
    #2. Compute covariance along time axis
    cov   =  npsum((x - xmean)*(y - ymean))/(n)
    
    #3. Compute correlation along time axis
    r_val   = cov/(xstd*ystd)

    if r_val > -0.995:
        return None, None, None

    #4. Compute regression slope and intercept:
    slope     = cov/(xstd**2)
    intercept = ymean - xmean*slope  
    
    return r_val, slope, intercept




def calc_integrals(price, y_hat, peak_tup):
    """Integral calc (Vectorized)"""

    slice_arr = np.array([[0,1,0,1],[1,-1,2,3]])
    
    #slice_tup = ((0,1,0,1), (1,-1,2,3))

    res_arr = np.zeros(4, dtype=np.float64)

    for tup in slice_arr:

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
        
        res_arr[tup[2]] = result[0]
        res_arr[tup[3]] = result[1]

    return res_arr[0], res_arr[1], res_arr[2], res_arr[3]   




def extract_data_for_plotting(close, index, final_trendline, x_peaks, peak_tup, length_list, y_peak_list):
    if final_trendline is None: 
        return None


    y_peaks = [close[peak] for peak in x_peaks]

    length_list.append(int(final_trendline.length))
    y_peak_list.append(max(y_peaks))
    #print(f'max length: {max(length_list)}')
    #print(f'min y price: {min(y_peak_list)}')
    #print(f'max y price:{max(y_peak_list)}')


    scatter_act_peak = [close[i] if i in peak_tup else np.nan for i in range(len(index))]
    scatter = [close[i] if i in x_peaks else np.nan for i in range(len(index))]


    return (scatter, scatter_act_peak) 



def plot_final_peaks_and_final_trendline(df, tup_data, y_hat, timestamp, peak_tup, candidates_df, fit_plot=0):

    if tup_data is None: 
        return None
    

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


    fig, axlist = fplt.plot(df_slice, figratio=(16,9), type='line', style='binance', title='Trend Hunter - ETHUSDT - 15M', alines=dict(alines=trendl_plot) , addplot=subplt, ylabel='Price ($)', returnfig=True, savefig=f'{path}/{str(timestamp)}.png')
    #fig, axlist = fplt.plot(df_slice, figratio=(16,9), type='candle', style='binance', title='Trend Hunter - ETHUSDT - 15M', ylim=(1100.0, 1650.0), alines=dict(alines=trendl_plot) , addplot=subplt, ylabel='Price ($)', returnfig=True, savefig=f'{path}/{str(timestamp)}.png')

    #df.reset_index(inplace=True)

    #plt.scatter(df_slice.index, scatter_slice, c='green')
    #plt.plot(df_slice.index, y_hat_slice, color='blue')
    #plt.plot(df_slice.Close, '-')
    #plt.title('Trend Hunter - ETHUSDT - 1D')
    #plt.grid()

    #fplt.show()

    to_json(df_slice, y_hat_slice)

    return df.index[peak_tup[-1]]



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