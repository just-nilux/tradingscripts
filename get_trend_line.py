from scipy.signal import find_peaks_cwt
from itertools import combinations
from scipy.stats import linregress
import pandas as pd
import numpy as np


def detect_peaks(df):

    x_peaks = find_peaks_cwt(df.Close, widths=np.arange(5, 15))

    return x_peaks


def all_combi_af_peaks(x_peaks):

    x_peak_combinations_list = list()

    sample_set = set(x_peaks)

    for n in range(len(sample_set) +1):

        x_peak_combinations_list += list(combinations(sample_set, 3))

    x_peak_combinations_list = list(dict.fromkeys(x_peak_combinations_list))
    
    return x_peak_combinations_list


def fetch_y_values_peaks(df, x_peak_combinations_list):

    y_peaks_combination_list = list()
    
    for y in x_peak_combinations_list:

        x1 = df.iloc[y[0]].Close
        x2 = df.iloc[y[1]].Close
        x3 = df.iloc[y[2]].Close
        temp = (x1, x2, x3)
        y_peaks_combination_list.append(temp)


    return y_peaks_combination_list


def peak_regression(x_peak_combinations_list, y_peaks_combination_arr):

    # zip de to x / y peaks array sammen inden regression:
    #x_y_peak_combi_arr = [item for item in zip(x_peak_combinations_list, y_peaks_combination_arr)]

    peak_regression_list = list()

    for i in range(len(x_peak_combinations_list)):

        slope, intercept, r_value, p_value, std_err  = linregress(x_peak_combinations_list[i], y_peaks_combination_arr[i])
        x = (slope, intercept, r_value, p_value, std_err )
        peak_regression_list.append(x)

    return peak_regression_list








    


    