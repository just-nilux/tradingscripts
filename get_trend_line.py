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


def remove_to_close_peaks(x_peaks):

    temp = list()

    for i, peak in enumerate(x_peaks):
        if i==0: temp.append(peak)
        elif abs(temp[-1]-peak) < 6:
            continue
        else: 
            temp.append(peak)
    
    return temp 


def detect_peaks_guassian(df):

    if df.empty: return
    
    df.reset_index(inplace=True)

    peak_list = list()

    dataFiltered = gaussian_filter1d(df.Close, sigma=2)
    x_peaks = signal.argrelmax(dataFiltered)[0] #order = 5 ?
    x_peaks = remove_to_close_peaks(x_peaks)
    
    print(x_peaks)
    #if len(x_peaks) > 3 and len(x_peaks) < 10:
    peak_list.append(x_peaks)
    
    if len(peak_list) > 0:
        return min(peak_list,key=len)


def detect_peaks(df):
    
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


            if len(x_peaks) > 5 and len(x_peaks) < 10:
                peak_list.append(x_peaks)

    
    print(f'tops: {min(peak_list,key=len)}')

    
    return min(peak_list,key=len)
    



def tune_peaks(df, x_peaks):

    # get the highest value previous 10 candles / 10 future candles from peak

    x_peaks_np = list()

    df['enumerate'] = df.index

    for peak in x_peaks:

        previous = peak -10
        forward = peak + 10

        highest_price = df.loc[previous:forward, 'Close'].idxmax()

        x_peaks_np.append(highest_price)

   
    return x_peaks_np


        

def all_combi_af_peaks(x_peaks):

    x_peaks_combinations_list = list()

    x_peaks_combinations_list = list(combinations(x_peaks, 3))
    x_peaks_combinations_list.sort(key=lambda tup: tup[1])
    x_peaks_combinations_list = list(dict.fromkeys(x_peaks_combinations_list))
    
    ([x_peaks_combinations_list.pop(a) for a, i in enumerate(x_peaks_combinations_list) if i[0]==i[1]==i[2]])
    
    assert all(len(tup) == 3 for tup in x_peaks_combinations_list)

    print(f'Antal trendl kombi: {len(x_peaks_combinations_list)}')
    

    return x_peaks_combinations_list



def fetch_y_values_peaks(df, x_peaks_combinations_list):

    y_peaks_combination_list = list()
    
    for y in x_peaks_combinations_list:

        x1 = df.iloc[y[0]].Close
        x2 = df.iloc[y[1]].Close
        x3 = df.iloc[y[2]].Close
        temp = (x1, x2, x3)
        y_peaks_combination_list.append(temp)
    
    return y_peaks_combination_list



def peak_regression(x_peak_combinations_list, y_peaks_combination_arr):


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

    return trendl_candidates_df



def fetch_trendl_start_end_price(df, trendl_candidate_df):
    
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

    return trendl_candidate_df



def trendline_angle_degree(trendl_candidates_df):

    trendl_candidates_df['angle_degree'] = trendl_candidates_df.slope.apply(lambda x: math.degrees(math.atan(x)))

    trendl_candidates_df.drop(trendl_candidates_df[trendl_candidates_df.angle_degree > 0].index, inplace=True)
    trendl_candidates_df.drop(trendl_candidates_df[trendl_candidates_df.r_value == -1.0].index, inplace=True)


    return trendl_candidates_df



def area_under_trendl():
    pass




def plot_all_trendl(df, trendl_candidates, x_peaks):
    

    # plot peaks:

    y_peaks = list()
    x_peaks_date = list()

    for peak in x_peaks:   
        y_peaks.append(df.iloc[peak].Close)
        x_peaks_date.append(df.iloc[peak].Date)
    
    plt.scatter(x_peaks, y_peaks, c='green')
    

    # plot alle mulige trendlines:

    for row in trendl_candidates.iterrows():
        slope = row[1].slope
        intercept = row[1].intercept
        y_hat = slope*df.index + intercept

        #plt.plot(df.index, y_hat, color='blue')
    
    print(np.trapz(y_peaks, x=x_peaks)) # Areal


    # Plot best line:
    #trendl_candidates['r_value'].idxmax()

    #print(trendl_candidates)
    if trendl_candidates.r_value.iloc[0] == -1:
        best_trendl = trendl_candidates.iloc[2]
    else:
        best_trendl = trendl_candidates.iloc[0]

    y_hat_best = best_trendl.slope*df.index + best_trendl.intercept
    plt.plot(df.index, y_hat_best, color='blue')


    # plot df.Close:

    #df.set_index(df.Date, inplace=True)
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
    trendl_plot = list(zip(df.index, y_hat_best))


    ap = fplt.make_addplot(df['scatter'],type='scatter', markersize=70, color='blue')
    fig, axlist = fplt.plot(df, figratio=(16,9), type='candle', style='binance', title='Trend Hunter - ETHUSDT - 15M', alines=dict(alines=trendl_plot) , addplot=ap,  ylabel='Price ($)', volume=True, returnfig=True)

    
    fplt.show()




def main():

    #path = '//home/traderblakeq/Python/klines/ETHUSDT1D.pkl'
    #os.chdir(path)

    df = pd.read_pickle('ETHUSDT15M.pkl').loc['2022-11-09' : '2022-11-10 01:00:00']  #.loc['2022-11-09' : '2022-11-10 01:00:00']

    # '2019-06-24' :'2019-12-23'

    df.reset_index(inplace=True)

    x_peaks = detect_peaks_guassian(df)

    #x_peaks = detect_peaks(df)

    x_peaks = tune_peaks(df, x_peaks)

    x_peaks_combinations_list = all_combi_af_peaks(x_peaks)
    
    y_peaks_combination_list = fetch_y_values_peaks(df, x_peaks_combinations_list)
    
    trendl_candidates = peak_regression(x_peaks_combinations_list, y_peaks_combination_list)

    trendl_candidates = fetch_trendl_start_end_price(df, trendl_candidates)

    trendl_candidates = trendline_angle_degree(trendl_candidates)

    plot_all_trendl(df, trendl_candidates, x_peaks)


    return df, trendl_candidates


#if __name__=='__main__':
#    import sys

#    if len(sys.argv) == 2:
#        df = sys.argv[:1]
#        main(df)






    


    