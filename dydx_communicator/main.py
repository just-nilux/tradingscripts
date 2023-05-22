from strategies.double_bottom_detector import DoubleBottomDetector
from strategies.double_top_detector import DoubleTopDetector
from DydxClient import DydxClient
from json_file_processor import process_json_file

import pandas_ta as ta
import pandas as pd
import datetime
import logging
import json
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')




def timeframe_to_minutes(timeframe):
    units = {
        "MIN": 1,
        "MINS": 1,
        "HOUR": 60,
        "HOURS": 60,
        "DAY": 1440,
    }

    # Split the input string into numbers and text parts
    timeframe_value = int(''.join(filter(str.isdigit, timeframe)))
    timeframe_unit = ''.join(filter(str.isalpha, timeframe))

    return timeframe_value * units[timeframe_unit]



def initialize_detectors(client):
    detectors = {}
    for strategy in client.config['strategies']:
        symbols = strategy['symbols']
        timeframes = strategy['timeframes']
        strategy_functions = strategy['strategy_functions']

        for symbol in symbols:
            for timeframe in timeframes:
                for strategy_function in strategy_functions:
                    key = f"{symbol}_{timeframe}_{strategy_function}"
                    
                    if strategy_function == "double_bottom_strat":
                        detectors[key] = DoubleBottomDetector(n_periods_to_confirm_swing=5, invalidation_n=72)
                    elif strategy_function == "double_top_strat":
                        detectors[key] = DoubleTopDetector(n_periods_to_confirm_swing=5, invalidation_n=72)
                    else:
                        logging.error(f"Unsupported strategy function: {strategy_function}")
                        continue

                    logging.debug(f"Initialized detector for {key}")

    return detectors




def double_top_strat(df, detector, ressist_zone_upper, ressist_zone_lower):

    logging.debug(f"Executing double top strategy for detector {detector}")

    detector.resistance_zone_upper = ressist_zone_upper
    detector.resistance_zone_lower = ressist_zone_lower
    detector.current_row = df.iloc[-1]
    side = detector.detect()

    if isinstance(side, tuple) and side[1] == 'SELL':
        return side
    return (None, None)



def double_bottom_strat(df, detector, support_zone_upper, support_zone_lower):

    logging.debug(f"Executing double bottom strategy for detector {detector}")

    detector.support_zone_upper = support_zone_upper
    detector.support_zone_lower = support_zone_lower
    detector.current_row = df.iloc[-1]
    res = detector.detect()

    if isinstance(res, tuple) and res[1] == 'BUY':
        return res
    return (None, None)



def fetch_support_resistance(symbol):
    # implement sup / rest levels from .json:
    # Dummy data for now...

    print(symbol)
    support_upper = 100000
    support_lower = 100
    resistance_upper = 120000
    resistance_lower = 110

    return support_upper, support_lower, resistance_upper, resistance_lower




def execute_strategies(client, detectors, liq_levels):
    current_time = datetime.datetime.now()
    minutes = current_time.minute

    for strategy in client.config['strategies']:
        symbols = strategy['symbols']
        timeframes = strategy['timeframes']

        for symbol in symbols:
            for timeframe in timeframes:
                timeframe_minutes = timeframe_to_minutes(timeframe)

                if not (minutes % timeframe_minutes):
                    # fordi at [:-1] er nyeste ikke lukket candle.
                    df = client.get_klines(symbol, timeframe)[:-1]
                    
                    # Calculate support and resistance levels
                    support_upper, support_lower, resistance_upper, resistance_lower = fetch_support_resistance(symbol)
                    
                    for strategy_function_name in strategy['strategy_functions']:
                        strategy_function = globals()[strategy_function_name]
                        detector_key = f"{symbol}_{timeframe}_{strategy_function_name}"
                        detector = detectors[detector_key]


                        try:
                            logging.debug(f"Executing strategy {strategy_function_name} for {symbol} on {timeframe}")
                            if strategy_function_name == "double_bottom_strat":
                                signal = strategy_function(df, detector, support_zone_upper=support_upper, support_zone_lower=support_lower)


                            elif strategy_function_name == "double_top_strat":
                                signal = strategy_function(df, detector, ressist_zone_upper=resistance_upper, ressist_zone_lower=resistance_lower)

                        except Exception as e:
                            print(f"Error while executing strategy for {symbol} on {timeframe}: {e}")
                            logging.error(f"Error while executing strategy for {symbol} on {timeframe}: {e}")
                       
                        try:
                            
                            if signal[1] in ('SELL', 'BUY'):
                                logging.debug(f"Executing signal {strategy_function_name} for {symbol} on {timeframe} - side: {signal[1]}")

                                size = float(client.order_size(symbol, client.config['position_size']))
                                logging.debug(f"Placing {signal[1].lower()} order for {symbol} with size {size}")
                                
                                atr = ta.atr(df.High, df.Low, df.Close, length=14).iloc[-1]
                                order = client.place_market_order(symbol=symbol, size=size, side=signal[1], atr=atr, trigger_candle=signal[0])

                            elif signal[1] is None:
                                logging.info(f"No signal for symbol: {symbol} on TF: {timeframe} - {strategy_function_name}")
                            else:
                                logging.warning(f"Invalid signal: {signal[1]}")

                        except Exception as e:
                            logging.error(f"Error while executing signal for {symbol} on {timeframe}: {e} - side: {signal[1]}")






def main():
    logging.info("Initializing detectors")
    client = DydxClient()
    detectors = initialize_detectors(client)
    
    # .json filepath:
    json_file_path = '/opt/tvserver/database.json'
    
    # Initialize the last hash as an empty string
    process_json_file.last_hash = ''

    while True:
        # Get the current time
        current_time = datetime.datetime.now()

        # Calculate the remaining seconds until the next minute
        remaining_seconds = 60 - current_time.second

        # Sleep for the remaining seconds
        time.sleep(remaining_seconds)

        liq_levels = process_json_file(json_file_path)
        execute_strategies(client, detectors, liq_levels)

        # Sleep for some time before executing the strategies again (e.g., 60 seconds)
        logging.debug("Sleeping untill next minute")
        
        # Sleep for 1 second to ensure it runs at the beginning of the minute
        time.sleep(1)


if __name__ == '__main__':
    main()

