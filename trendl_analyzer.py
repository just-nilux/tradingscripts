import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd
import numpy as np
import math



def fetch_y_values_peaks(price , x_touches):
    """
    Return max(df.Close, df.Open) at each peak in peak combinations list.
    :params x_peak_combinations
        List of combinations of length 3.
    """
    if x_touches is None: 
        return
    
    y_touches1 = price[x_touches[0]]
    y_touches2 = price[x_touches[1]]
    y_touches3 = price[x_touches[2]]


    y_touches1 = price.iloc[x_touches[0]]

    print(y_touches1)
   
    y_touches = np.array([y_touches1, y_touches2, y_touches3])

    return y_touches




def peak_regression(price, x_touches, y_touches):
    """
    :param x_peak_combinations
        List of peak index combinations (tuples) of len 3
    :param x_peak_combinations
        List of peak value combinations (tuples) of len 3
    """
    if x_touches is None:
        return
    

    slope, intercept, r_value, p_value, std_err  = linregress(x_touches, y_touches, alternative='less')

        
    peak_tup = tuple(x_touches)
    y_hat = slope*np.arange(0, len(price)) + intercept
    aboveArea_p1_p2, belowArea_p1_p2, aboveArea_p2_p3, belowArea_p2_p3 = calc_integrals(price, y_hat, peak_tup)

    print()
    print(f'Trendline length: {x_touches[-1] - x_touches[0]}')
    print(f'angle degree: {math.degrees(math.atan(slope))}')
    print(f'r_val: {r_value}')
    print()
    print(f'Above Area (P1P2): {aboveArea_p1_p2}')
    print(f'Below Area (P1P2): {belowArea_p1_p2}')
    print()
    print(f'Above Area (P2P3): {aboveArea_p2_p3}')
    print(f'Below Area (P2P3): {belowArea_p2_p3}')

    return y_hat
    



def calc_integrals(price, y_hat, peak_tup, details=None):

    y1 =price[peak_tup[0]:peak_tup[1]+1]
    y2 =y_hat[peak_tup[0]:peak_tup[1]+1]
    x1 = np.arange(0, len(y1))

    y3 =price[peak_tup[1]:peak_tup[-1]+1]
    y4 =y_hat[peak_tup[1]:peak_tup[-1]+1]
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


def plotter(df, y_hat, x_touches, y_touches):

    df.reset_index(inplace=True)

    plt.scatter(x_touches, y_touches, c='green')
    plt.plot(df.index, y_hat, color='blue')
    plt.plot(df.Close, '-')
    plt.title('Trend Analyzer')
    plt.grid()

    plt.show()


def main(start, stop, x_touches):
    """
    Simple script to analyse Simons trendlines.
       
    example: 
       
       main(start, stop, touches)

       main('2021-12-01', '2021-12-24', [4,15,24])

    """
    

    df = pd.read_pickle('./data/ETHUSDT15M.pkl')
    df.drop(['Close_time', 'Volume'], axis=1, inplace=True)
    df = df.loc[str(start):str(stop)]
    C = np.array(df.Close)
    print(df)


    y_touches = fetch_y_values_peaks(df.Close, np.array(x_touches))
    y_hat = peak_regression(C, x_touches, y_touches)
    plotter(df, y_hat, x_touches, y_touches )
