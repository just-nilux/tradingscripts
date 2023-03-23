from backtesting import Backtest, Strategy
from backtesting.lib import crossover, cross, resample_apply
import pandas as pd
from strategies import *
import zigzag

import pandas as pd


def detect_extreme_rsi(data):
    """
    Detects extreme cycles of the Relative Strength Index (RSI) for a pandas Series of data, 
    where the RSI goes above a high threshold of 75. Returns a pandas DataFrame with an 
    additional column 'extreme_rsi', where True indicates the row with the maximum RSI value 
    for each extreme cycle. Uses the pandas_ta library to calculate the RSI.
    """
    
    # calculate RSI using the ta library
    rsi = ta.momentum.rsi(data['Close'], 14)
    data['RSI_HTF'] = rsi

    # define the RSI extreme thresholds
    high_threshold = 75

    # find the RSI extreme cycles (i.e., when RSI goes above/below the thresholds)
    high_extreme_rsi = (rsi > high_threshold).astype(int)

    # create a list of lists to store the RSI values and their corresponding indices
    rsi_values = []
    last_j = data.index[0]
    for i, val in high_extreme_rsi.iteritems():
        if i < last_j:
            continue
        if val == 1:
            # create a new list to store the RSI values and their corresponding indices for this cycle
            cycle_rsi = [[i, rsi[i]]]
            # loop through the rest of the cycle until RSI crosses below the high threshold
            j = i
            while True:
                j += pd.Timedelta(data.index.to_series().diff()[1])
                if j > high_extreme_rsi.index[-1]:
                    break
                if high_extreme_rsi[j] == 0:
                    cycle_rsi.append([j, rsi[j]])
                    rsi_values.append(cycle_rsi)
                    last_j = j
                    break
    
    extreme_rsi = pd.Series(False, index=data.index)
    for cycle in rsi_values:
        extreme_rsi.loc[rsi.loc[cycle[0][0]:cycle[1][0]].idxmax()] = True
    
    data = pd.concat([data, pd.Series(extreme_rsi, index=data.index, name='extreme_rsi')], axis=1)
    
    return data


def merge_df(LTF, HTF):

    merged_df = LTF.merge(HTF, on =['Date'], how='left').ffill()
    merged_df.rename(columns={"Open_x":"Open", "High_x":"High", "Low_x":"Low", 'Close_x':'Close', 'Volume_x':'Volume', 'Open_y':'Open_HTF', 'High_y':'High_HTF', 'Low_y':'Low_HTF', 'Close_y':'Close_HTF', 'Volume_y':'Volume_HTF'}, errors="raise", inplace=True)
    merged_df = merged_df.dropna()
    return merged_df

def zigz(Close, up_threshold, down_threshold, remove_duplicates=False):

    #Close = Close.drop_duplicates(keep='first')
    if remove_duplicates:    
        Close = Close[Close.index.minute == 0]

    zig = zigzag.peak_valley_pivots(Close, up_threshold, down_threshold)
    #plot_zig(df.Close, zig)
    Close = pd.concat([Close, pd.Series(zig, index=Close.index, name='zig')], axis=1)
    Close = Close.iloc[1:-1]
    swing = Close.where(Close["zig"] != 0)
    swing = swing[swing['zig'].notna()]
    return swing


def rsi_bear_div_cycle_detected(self):

    if not self.position and self.data.index.minute[-1] == 0 and self.data.extreme_rsi and self.setup_invalidated:
        return True
    else:
        return False




def reset(self):

    self.date_price_at_rsi_extreme = tuple()
    self.formation_list = list()
    self.bear_div_cycle_started = False
    self.is_first_time = True
    self.price_HTF_prev_elbow = 0
    self.entry_requirements_forfilled = False


class Zigzag(Strategy):
    
    prev_zigzag_list = list()
    date_price_at_rsi_extreme = tuple()
    formation_list = list()
    bear_div_cycle_started = False
    is_first_time = True
    price_HTF_prev_elbow = 0
    entry_requirements_forfilled = False
    setup_invalidated = True


    
    def init(self):
        
        self.df = self.data.df.copy()
        
    
    def next(self):

        rsi = self.data.RSI_HTF[-1]
        price_HTF = self.data.Close_HTF[-1]
        price_LTF = self.data.Close[-1]
        index = self.data.index[-1]
            
        # if rsi bear div cycle detected:
        if rsi_bear_div_cycle_detected(self):
            self.date_price_at_rsi_extreme = ('bear_div', index, price_HTF, rsi)
            print(f'RSI extreme detected: {self.date_price_at_rsi_extreme[1]} - Price: {self.date_price_at_rsi_extreme[2]} - RSI: {self.date_price_at_rsi_extreme[3]}')
            self.bear_div_cycle_started = True
            self.setup_invalidated = False


        if not self.bear_div_cycle_started or self.position:
            return
        

        # if rsi > than at rsi extreme = no setup:
        if rsi > self.date_price_at_rsi_extreme[3]: #and rsi<75:
            print(f'setup invalidated by HH on RSI - {index}')
            print('---------------------')
            self.setup_invalidated = True
            reset(self)
            return
        
        #elif rsi > self.date_price_at_rsi_extreme[3] and rsi>75 and price_LTF > self.date_price_at_rsi_extreme[2]:
            #reset(self)
        #    self.date_price_at_rsi_extreme = ('bear_div', index, price_HTF, rsi)
            #self.setup_invalidated = False
            #self.bear_div_cycle_started = False
        #    print(f'!RSI extreme detected: {self.date_price_at_rsi_extreme[1]} - Price: {self.date_price_at_rsi_extreme[2]} - RSI: {self.date_price_at_rsi_extreme[3]}')




        # Look for new HH in bear rsi div. setup:
        df = self.df.loc[self.date_price_at_rsi_extreme[1]:self.data.index[-1]].copy()
        swing = zigz(df.Close_HTF, 0.03, -0.03, remove_duplicates=True)
        
        if not swing.empty and swing.index[-1] != self.date_price_at_rsi_extreme[1] and swing.Close_HTF[-1] > self.date_price_at_rsi_extreme[2] and rsi<self.date_price_at_rsi_extreme[3]:
            
            self.date_price_at_rsi_extreme = ('bear_div', swing.index[-1], swing.Close_HTF[-1], self.df.loc[swing.index[-1]]['RSI_HTF'])
            print(f'New HH on Price & LL on RSI: {self.date_price_at_rsi_extreme}')


        # if in rsi bear div territory:
        if price_HTF>self.date_price_at_rsi_extreme[2] and rsi<self.date_price_at_rsi_extreme[3] or not self.is_first_time:

            # reset elbow ts:
            if not self.is_first_time and price_LTF > self.price_HTF_prev_elbow[0]:
                #print(f'if : price_ltf:  {price_LTF} + {self.data.index[-1]} > price at ext: {self.price_HTF_prev_elbow}')
                print(f'Elbow reset: {self.data.index[-1]}')
                self.is_first_time = True

            # Detect Elbow confirmation on RSI (HTF):
            if self.data.RSI_HTF < self.data.RSI_HTF[-2]:
                if self.is_first_time:
                    self.elbow_ts = index
                    self.price_HTF_prev_elbow = (self.data.Close_HTF[-2], self.data.index[-2])
                    self.is_first_time = False
                    print(f'elbow ts: {self.elbow_ts}')
                
            # Start entry strategi - kig efter LL på LTF (5/15m) = Short
            if not self.is_first_time: 
                self.df_slice_LTF = df.loc[self.elbow_ts:]
                swing_LTF = zigz(self.df_slice_LTF.Close , 0.003, -0.003, remove_duplicates=False)
                if len(swing_LTF) >=2:
                    if swing_LTF['zig'][-1] == 1 and swing_LTF['zig'][-2] == -1 and price_LTF < swing_LTF['Close'][-2]:
                        print(f'SHORT: {index} - {price_LTF}')
                        self.entry_requirements_forfilled = True

            # FIB implementering for TP:
            if self.entry_requirements_forfilled: # and self.data.Funding_rate[-1] > 0.010000:

                atr = self.data.ATR_HTF[-1]
                self.is_first_time = True
                print('be aware')
                try:
                    self.sell(tp=price_HTF-(atr*2), sl=price_HTF+(atr*1.5))
                except ValueError as e:
                    if str(e) == "StopLoss cannot be higher than TakeProfit.":
                        SL = self.df_slice_LTF.High.rolling(len(self.df_slice_LTF)).max()[-1]
                        self.sell(tp=price_HTF-atr*2, sl=SL)
                    else:
                        return
                

                
               

                





HTF = pd.read_pickle('./data/BTCUSDT1H.pkl')
LTF = pd.read_pickle('./data/BTCUSDT15M_fund.pkl')

HTF['ATR_HTF'] = ta.atr(HTF.High, HTF.Low, HTF.Close, length=14, percent=False)

HTF = detect_extreme_rsi(HTF)
HTF.dropna(inplace=True)

df = merge_df(LTF,HTF)

df = df.loc['2022-12-01':].copy()

print(df)


bt = Backtest(df, Zigzag,
              cash=1000000, 
              commission= .00,
              exclusive_orders=True,
              trade_on_close=False,
              hedging=False
              )

output = bt.run()
print(output)
bt.plot()



def lol():

    # Find first LL:
    selected_date = self.price_HTF_prev_elbow[1]
    df_slice = self.df.iloc[self.df.index.get_loc(selected_date) - 600: self.df.index.get_loc(selected_date) + 1, self.df.columns.get_loc('Close_HTF')]
    res = zigz(df_slice, 0.009, -0.009, remove_duplicates=True)                
    res = res[res['zig'] == -1.0]
    res = res.iloc[::-1]
    res['previous_Close_HTF'] = res['Close_HTF'].shift(-1)

    try:
        result = res[res['Close_HTF'] < res['previous_Close_HTF']].iloc[0]
    except IndexError:
        print("Index Error occurred. The condition res['Close_HTF'] < res['previous_Close_HTF'] may not be satisfied for any row in the dataframe.")
        result = res.iloc[0]

    
    # Find min / max hvor fib skal trækkes fra og til:
    high = df_slice.loc[selected_date] # Close value på candle før elbow.
    low =  df_slice.loc[result.name] # LL (Sidste swing) - finder den via zigzag

    # beregn pris range:
    range = high - low

    # Fibonacci retracement levels
    level_0 = high
    level_61_8 = high - range * 0.382
    level_50 = high - range * 0.5
    level_38_2 = high - range * 0.618
    level_100 = low
    
    TP = level_61_8
    SL = self.df_slice_LTF.High.rolling(len(self.df_slice_LTF)).max()[-1]
    print(f'Fib high: {selected_date}')
    print(f'Fib low: {result.name}')
    
    print(f'TP: {TP}')
    print(f'SL: {SL}')

    try:
        self.sell(tp=TP, sl=SL)
    except ValueError:
        print('###############################')
        self.sell(tp=level_100, sl=SL)
        print(level_100)
    #reset(self)