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





def detect_peaks_guassian(price, sigma=0.5):
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

    #tuned_peaks = tune_peaks(price, x_peaks)
    #cleaned_peaks = remove_too_close_peaks(tuned_peaks)


    if len(x_peaks) >= 3:
        return x_peaks

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



def peak_regression(price, x_peak_combinations, y_peak_combinations):
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
        negative = math.degrees(math.atan(slope))
        if r_value < -0.99 and negative < 0:
          
            peak_tup = tuple(x_peak_combinations[i])
            y_hat = slope*np.arange(0, len(price)) + intercept
            aboveArea_p1_p2, belowArea_p1_p2, aboveArea_p2_p3, belowArea_p2_p3 = calc_integrals(price, y_hat, peak_tup)

            if True: #aboveArea_p1_p2 <20 and belowArea_p1_p2 < 20 and aboveArea_p2_p3 < 500 and belowArea_p2_p3 < 500 and belowArea_p1_p2 > 100 and belowArea_p2_p3 > 100:
          
                df.loc[i, 'start_index'] = x_peak_combinations[i][0]
                df.loc[i, 'end_index'] = x_peak_combinations[i][2]
                df.loc[i, 'slope'] = slope
                df.loc[i, 'intercept'] = intercept
                df.loc[i, 'r_value'] = r_value
                df.loc[i, 'p_value'] = p_value
                df.loc[i, 'std_err'] = std_err
                #df.loc[i, 'aboveArea_p1_p2'] = aboveArea_p1_p2
                #df.loc[i, 'belowArea_p1_p2'] = belowArea_p1_p2
                #df.loc[i, 'aboveArea_p2_p3'] = aboveArea_p2_p3
                #df.loc[i, 'belowArea_p2_p3'] = belowArea_p2_p3                

                #assert len(trendl_candidates_df) != 0, \
                #   f'No Trendl candidates - peak_regression'
    
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



def trendline_angle_degree(candidates_df):
    if candidates_df is None: 
        return

    assert len(candidates_df) != 0, \
            f'No trendl candadates - (trendline_angle_degree)'

    candidates_df['angle_degree'] = candidates_df \
            .slope.apply(lambda x: math.degrees(math.atan(x)))

    if candidates_df.empty:
        return None
    else: 
        return candidates_df



def area_under_trendl():
    pass



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



def extract_data_for_plotting(close, index, final_trendline, x_peaks):
    if final_trendline is None: return

    assert len(final_trendline) != 0 , f'No trendl candidates - (extract_data_for_plotting)'

    # Save x,y peaks to list:
    y_peaks_date = list()
    y_peaks = list()
   
    for peak in x_peaks:
        y_peaks_date.append(index[peak])
        y_peaks.append(close[peak])


    scatter = np.full(len(index), fill_value=np.nan)

    for peak in x_peaks:
        scatter[peak] = close[peak]


    return (scatter, y_peaks_date, y_peaks) 



def plot_final_peaks_and_final_trendline(df, tup_data, x_peaks, y_hat, timestamp):

    if tup_data is None: return
    
    scatter, y_peaks_date, y_peaks = tup_data[0], tup_data[1], tup_data[2]

    
    trendl_plot = list(zip(df.index, y_hat))
   
    trendl_start_end = list([trendl_plot[0], trendl_plot[-1]])
    
            
    path = './trendline_results'
    
    ap = fplt.make_addplot(scatter,type='scatter', markersize=70, color='blue')
    fig, axlist = fplt.plot(df, figratio=(16,9), type='line', style='binance', title='Trend Hunter - ETHUSDT - 15M', alines=dict(alines=trendl_plot) , addplot=ap,  ylabel='Price ($)', returnfig=True, savefig=f'{path}/{str(timestamp)}.png')
    #fig, axlist = fplt.plot(df, figratio=(16,9), type='candle', style='binance', title='Trend Hunter - ETHUSDT - 15M', alines=dict(alines=trendl_plot) , addplot=ap,  ylabel='Price ($)', returnfig=True)
    
    df.reset_index(inplace=True)

    plt.scatter(x_peaks, y_peaks, c='green')
    plt.plot(df.index, y_hat, color='blue')
    plt.plot(df.Close, '-')
    plt.title('Trend Hunter - ETHUSDT - 1D')
    plt.grid()

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