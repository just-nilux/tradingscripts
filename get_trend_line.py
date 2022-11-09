from scipy.signal import find_peaks_cwt
from itertools import combinations
from scipy.stats import linregress
import pandas as pd
import numpy as np


def detect_peaks(df):

    x_peaks = find_peaks_cwt(df.Close, widths=np.arange(5, 15))

    #y_peaks = np.array([])

    #for i in x_peaks:
    #    y_peaks = np.append(y_peaks, df.iloc[i].Close)

    #peak_list = list(zip(x_peaks, y_peaks))

    return x_peaks


def all_combi_af_peaks(x_peaks):

    x_peak_combinations_list = np.array([])

    sample_set = set(x_peaks)

    for n in range(len(sample_set) +1):
        x_peak_combinations_list += list(combinations(sample_set, 3))

    x_peak_combinations_list = list(dict.fromkeys(x_peak_combinations_list))

    x_peak_combinations_arr = np.array(x_peak_combinations_list)
    
    return x_peak_combinations_arr


def fetch_y_values_peaks(df, x_peak_combinations_arr):

    y_peaks = np.array([])

    for x in x_peak_combinations_arr:
        for y in x:
            temp = list()
            temp.append(df.iloc[y[0]].Close)
            temp.append(df.iloc[y[1]].Close)
            temp.append(df.iloc[y[2]].Close)
            y_peaks = np.append(y_peaks, temp)

    return y_peaks


def peak_regression(peak_combinations_list):

    peak_regression_list = list()

    for combination in peak_combinations_list:

        slope, intercept, r_value, p_value, std_err  = linregress(combination[0], combination[1])






    


    