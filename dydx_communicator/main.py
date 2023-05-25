from strategies.double_bottom_detector import DoubleBottomDetector
from strategies.double_top_detector import DoubleTopDetector
from strategies.liq_sweep_detector import SweepDetector
from collections import defaultdict
from DydxClient import DydxClient
from json_file_processor import process_json_file
from send_telegram_message import bot_main, send_telegram_message
from logger_setup import setup_logger
from dydx_candle_retriever import get_all

import asyncio
import pandas_ta as ta
import pandas as pd
import threading
import datetime
import json
import time

logger = setup_logger(__name__)


def check_liquidation_zone(data, client):
    """
    Check the liquidation zone for each symbol in the data and send a 
    Telegram message if the current price is outside the provided zone.

    Args:
    data (dict): A dictionary with symbols as keys and list of prices as values.
    client (obj): Client object to connect with the server and get market data.
    """

    for symbol, prices in data.items():
        if len(prices) == 4:
            market = client.client.public.get_markets(market=symbol).data['markets'][symbol]
            current_price = float(market['oraclePrice'])
            min_price = min(prices)
            max_price = max(prices)

            if not min_price < current_price < max_price:
                msg = f'Update Liquidity Zones: {symbol}'
                send_telegram_message(client.config['bot_token'], client.config['chat_ids'], msg)
                logger.debug(msg)



def update_config_with_symbols(data: defaultdict, client):
    """
    Update the 'symbols' list in each strategy in the client's config with symbols from the provided defaultdict.                                                                            
    Symbols are selected from the defaultdict if their corresponding list has exactly 3 elements.                                                                                            
    The updated config is then written back to the client's config.json file.

    Parameters:
    data (defaultdict): The defaultdict containing symbol data. Keys are symbols, values are lists.                                                                                          
    client (object): The client object, expected to have 'config.json'.                                                                                              

    Returns: True if some symbols have been added, False otherwise
    """
    
    symbols_added = False

    # Update the symbols in strategies for symbols with length == 4 in defaultdict
    for strategy in client.config['strategies']:
        new_symbols = [k for k, v in data.items() if len(v) == 4]
        if set(new_symbols) != set(strategy['symbols']):
            strategy['symbols'] = new_symbols
            symbols_added = True

    # Write back the updated json to file
    if symbols_added:
        with open("config.json", 'w') as json_file:
            json.dump(client.config, json_file, indent=2)

    return symbols_added





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



def initialize_detectors(client, detectors=None):
    if detectors is None:
        detectors = {}

    for strategy in client.config['strategies']:
        symbols = strategy['symbols']
        timeframes = strategy['timeframes']
        strategy_functions = strategy['strategy_functions']

        for symbol in symbols:
            for timeframe in timeframes:
                for strategy_function in strategy_functions:
                    key = f"{symbol}_{timeframe}_{strategy_function}"
                    
                    if key not in detectors:
                        if strategy_function == "double_bottom_strat":
                            detectors[key] = DoubleBottomDetector(n_periods_to_confirm_swing=5, invalidation_n=72)
                        elif strategy_function == "double_top_strat":
                            detectors[key] = DoubleTopDetector(n_periods_to_confirm_swing=5, invalidation_n=72)
                        elif strategy_function == "liq_sweep_detector":
                            detectors[key] = SweepDetector(n_periods_to_confirm_sweep=5, cross_pct_threshold=0.2)
                        else:
                            logger.error(f"Unsupported strategy function: {strategy_function}")
                            continue

                        logger.debug(f"Initialized detector for {key}")
    return detectors



def double_top_strat(df, detector, ressist_zone_upper, ressist_zone_lower):

    logger.debug(f"Executing double top strategy for detector {detector}")

    detector.resistance_zone_upper = ressist_zone_upper
    detector.resistance_zone_lower = ressist_zone_lower
    detector.current_row = df.iloc[-1]
    res = detector.detect()

    if isinstance(res, tuple) and res[1] == 'SELL':
        return res
    return (None, None)



def double_bottom_strat(df, detector, support_zone_upper, support_zone_lower):

    logger.debug(f"Executing double bottom strategy for detector {detector}")

    detector.support_zone_upper = support_zone_upper
    detector.support_zone_lower = support_zone_lower
    detector.current_row = df.iloc[-1]
    res = detector.detect()

    if isinstance(res, tuple) and res[1] == 'BUY':
        return res
    return (None, None)


def liq_sweep_detector(df, detector, upper_liq_level, lower_liq_level):

    logger.debug(f"Executing Liq_sweep_detector strategy for detector {detector}")

    detector.upper_liq_level = upper_liq_level
    detector.lower_liq_level = lower_liq_level
    detector.current_row = df.iloc[-1]
    res = detector.detect()

    if isinstance(res, tuple) and res[1] in ('BUY', 'SELL'):
        return res
    return (None, None)



def fetch_support_resistance(symbol, liq_levels):

    # fetch support and resistance levels
    liq_data = liq_levels[symbol]  # Retrieve the list of values for 'symbol'

    # Assign the values to variables
    support_lower = min(liq_data)
    support_upper = sorted(liq_data)[1]  # Second lowest value

    resistance_upper = max(liq_data)
    resistance_lower = sorted(liq_data)[-2]  # Second highest value


    return support_upper, support_lower, resistance_upper, resistance_lower




def execute_strategies(client, detectors, liq_levels, all_symbol_df):
    current_time = datetime.datetime.now()
    minutes = current_time.minute

    for strategy in client.config['strategies']:
        symbols = strategy['symbols']
        timeframes = strategy['timeframes']

        for symbol in symbols:
            for timeframe in timeframes:
                timeframe_minutes = timeframe_to_minutes(timeframe)

                if not (minutes % timeframe_minutes):

                    df = all_symbol_df[(symbol, timeframe)]

                    # fetch support and resistance levels
                    support_upper, support_lower, resistance_upper, resistance_lower = fetch_support_resistance(symbol, liq_levels)
                    
                    
                    for strategy_function_name in strategy['strategy_functions']:
                        strategy_function = globals()[strategy_function_name]
                        detector_key = f"{symbol}_{timeframe}_{strategy_function_name}"
                        detector = detectors[detector_key]


                        try:
                            logger.debug(f"Executing strategy {strategy_function_name} for {symbol} on {timeframe}")
                            if strategy_function_name == "double_bottom_strat":
                                signal = strategy_function(df, detector, support_zone_upper=support_upper, support_zone_lower=support_lower)

                            elif strategy_function_name == "double_top_strat":
                                signal = strategy_function(df, detector, ressist_zone_upper=resistance_upper, ressist_zone_lower=resistance_lower)
                            
                            elif strategy_function_name == "liq_sweep_detector":
                                signal = strategy_function(df, detector, upper_liq_level=resistance_upper, lower_liq_level=support_lower )

                        except Exception as e:
                            print(f"Error while executing strategy for {symbol} on {timeframe}: {e}")
                            logger.error(f"Error while executing strategy for {symbol} on {timeframe}: {e}")
                       
                        try:
                            
                            if signal[1] in ('SELL', 'BUY'):
                                logger.debug(f"Executing signal {strategy_function_name} for {symbol} on {timeframe} - side: {signal[1]}")

                                size = float(client.order_size(symbol, client.config['position_size']))
                                logger.debug(f"Placing {signal[1].lower()} order for {symbol} with size {size}")
                                
                                atr = ta.atr(df.High, df.Low, df.Close, length=14).iloc[-1]
                                order = client.place_market_order(symbol=symbol, size=size, side=signal[1], atr=atr, trigger_candle=signal[0])

                            elif signal[1] is None:
                                logger.info(f"No signal for symbol: {symbol} on TF: {timeframe} - {strategy_function_name}")
                            else:
                                logger.warning(f"Invalid signal: {signal[1]}")

                        except Exception as e:
                            logger.error(f"Error while executing signal for {symbol} on {timeframe}: {e} - side: {signal[1]}")




def execute_main(client, json_file_path, liq_levels, detectors):
    detectors = initialize_detectors(client)

    # Initialize the last hash as an empty string
    process_json_file.last_hash = ''

    while True:
        # Get the current time
        current_time = datetime.datetime.now()

        # Calculate the remaining seconds until the next minute
        remaining_seconds = 60 - current_time.second

        # Sleep for the remaining seconds
        time.sleep(remaining_seconds)

        res = process_json_file(json_file_path)

        if res is not None:
            liq_levels = res
            symbols_added = update_config_with_symbols(liq_levels, client)
            if symbols_added:
                detectors = initialize_detectors(client, detectors)

        # create unique sets of all symbols and timeframes
        all_symbols = set(symbol for strategy in client.config['strategies'] for symbol in strategy['symbols'])
        all_timeframes = set(timeframe for strategy in client.config['strategies'] for timeframe in strategy['timeframes'])

        # fetch symbol data
        all_symbol_df = asyncio.run(get_all(all_symbols, all_timeframes))

        execute_strategies(client, detectors, liq_levels, all_symbol_df)

        check_liquidation_zone(liq_levels, client)

        # Sleep for some time before executing the strategies again (e.g., 60 seconds)
        logger.debug("Sleeping untill next minute")

        # Sleep for 1 second to ensure it runs at the beginning of the minute
        time.sleep(1)


def main():
    logger.info("Initializing detectors")
    client = DydxClient()

    # Start the bot in a separate thread
    bot_thread = threading.Thread(target=execute_main, args=(client, '/opt/tvserver/database.json', defaultdict(list), initialize_detectors(client)))
    bot_thread.start()

    # Run the bot in the main thread
    bot_main(client.config['bot_token'], client)


if __name__ == '__main__':
    main()


