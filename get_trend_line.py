from scipy.signal import find_peaks_cwt
from scipy.stats import linregress
from itertools import combinations
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import os


def detect_peaks(df):

    x_peaks = find_peaks_cwt(df.Close, widths=np.arange(5, 15))

    return x_peaks


def all_combi_af_peaks(x_peaks):

    x_peaks_combinations_list = list()

    sample_set = set(x_peaks)

    for n in range(len(sample_set) +1):

        x_peaks_combinations_list += list(combinations(sample_set, 3))

    x_peaks_combinations_list = list(dict.fromkeys(x_peaks_combinations_list))
    
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

    peak_regression_list = list()


    for i in range(len(x_peak_combinations_list)):

        slope, intercept, r_value, p_value, std_err  = linregress(x_peak_combinations_list[i], y_peaks_combination_arr[i])
        x = (x_peak_combinations_list[i][0], x_peak_combinations_list[i][2], slope, intercept, r_value, p_value, std_err )
        peak_regression_list.append(x)

    trendl_candidates_df= pd.DataFrame(peak_regression_list,columns =['df_start_index', 'df_end_index' ,'slope', 'intercept', 'r_value', 'p_value', 'std_err'])

    trendl_candidates_df.sort_values('r_value', inplace=True)

    return trendl_candidates_df


def fetch_trendl_start_end_price(df, trendl_candidate_df):
    
    start_price = list()

    for row in trendl_candidate_df.df_start_index:
        print(df.iloc[row].Close)
        
        start_price.append(df.iloc[row].Close)
    
    trendl_candidate_df['start_price'] = start_price

    end_price = list()

    for row in trendl_candidate_df.df_end_index:
        
        end_price.append(df.iloc[row].Close)

    trendl_candidate_df['end_price'] = end_price

    return trendl_candidate_df


def trendline_angle_degree(trendl_candidates_df):

    trendl_candidates_df['angle_degree'] = trendl_candidates_df.slope.apply(lambda x: math.degrees(math.atan(x)))

    return trendl_candidates_df


def area_under_trendl():
    pass


def plot(df, trendl_candidates):
    pass



def main():

    #path = '//home/traderblakeq/Python/klines/ETHUSDT1D.pkl'
    #os.chdir(path)

    df = pd.read_pickle('ETHUSDT1D.pkl').loc['2022-04-03' :'2022-10-19']

    print(df)

    df.reset_index(inplace=True)

    x_peaks = detect_peaks(df)

    x_peaks_combinations_list = all_combi_af_peaks(x_peaks)
    
    y_peaks_combination_list = fetch_y_values_peaks(df, x_peaks_combinations_list)
    
    trendl_candidates = peak_regression(x_peaks_combinations_list, y_peaks_combination_list)

    trendl_candidates = fetch_trendl_start_end_price(df, trendl_candidates)

    trendl_candidates = trendline_angle_degree(trendl_candidates)


    return df, trendl_candidates


#if __name__=='__main__':
#    import sys

#    if len(sys.argv) == 2:
#        df = sys.argv[:1]
#        main(df)






    


    