from set_strategy_entry_obj import doubleTopEntry, doubleBottomEntry, liqSweepEntry
from strategies.doubleBottomEntry import DoubleBottomDetector
from strategies.doubleTopEntry import DoubleTopDetector
from strategies.liqSweepEntry import SweepDetector
from position_storage import PositionStorage
from collections import defaultdict
from dydxClient import DydxClient
from json_file_processor import process_json_file
from send_telegram_message import bot_main, send_telegram_message
from logger_setup import setup_logger
from dydx_candle_retriever import get_all
from talipp.indicators import ATR


import asyncio
import pandas_ta as ta
import pandas as pd
import threading
import datetime
import json
import time

logger = setup_logger(__name__)


def check_liquidation_zone(data: dict, client: DydxClient, liq_zones_to_be_updated: list):
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

            if min_price < current_price < max_price and symbol in liq_zones_to_be_updated:
                msg = f"\033[92m{chr(0x2713)}\033[0m Liquidity Zones have been updated for {symbol}"
                send_telegram_message(client.config['bot_token'], client.config['chat_ids'], msg)
                liq_zones_to_be_updated.remove(symbol)

            elif not min_price < current_price < max_price:
                liq_zones_to_be_updated.extend(symbol)
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
    new_symbols = []

    # Update the symbols in strategies for symbols with length == 4 in defaultdict
    for strategy in client.config['strategies']:
        new_symbols = [k for k, v in data.items() if len(v) == 4]
        if set(new_symbols) != set(strategy['symbols']):
            strategy['symbols'] = new_symbols
            symbols_added = True
            new_symbols.extend(new_symbols)

    # Write back the updated json to file
    if symbols_added:
        with open("config.json", 'w') as json_file:
            json.dump(client.config, json_file, indent=2)

    return new_symbols





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



def initialize_detectors(client, detectors=None, atrs=None):
    if detectors is None:
        detectors = {}

    if atrs is None:
        atrs = {}

    for strategy in client.config['strategies']:
        symbols = strategy['symbols']
        timeframes = strategy['timeframes']
        strategy_functions = strategy['strategy_functions']

        for symbol in symbols:
            for timeframe in timeframes:
                key = f"{symbol}_{timeframe}"

                if key not in atrs:
                    atrs[key] = ATR(14)

                for strategy_function in strategy_functions:
                    key = f"{symbol}_{timeframe}_{strategy_function}"
                    
                    if key not in detectors:
                        if strategy_function == "doubleBottomEntry":
                            detectors[key] = DoubleBottomDetector(n_periods_to_confirm_swing=5, invalidation_n=72)
                        elif strategy_function == "doubleTopEntry":
                            detectors[key] = DoubleTopDetector(n_periods_to_confirm_swing=5, invalidation_n=72)
                        elif strategy_function == "liqSweepEntry":
                            detectors[key] = SweepDetector(n_periods_to_confirm_sweep=5, cross_pct_threshold=0.2)
                        else:
                            logger.error(f"Unsupported strategy function: {strategy_function}")
                            continue

                        logger.debug(f"Initialized detector for {key}")

    # Cleanup step
    current_symbols = set(symbol for strategy in client.config['strategies'] for symbol in strategy['symbols'])
    current_timeframes = set(timeframe for strategy in client.config['strategies'] for timeframe in strategy['timeframes'])
    
    keys_to_delete = []
    for key in detectors.keys():
        symbol, timeframe, _ = key.split("_")
        if symbol not in current_symbols or timeframe not in current_timeframes:
            keys_to_delete.append(key)

    for key in keys_to_delete:
        del detectors[key]
        logger.debug(f"Removed detector for {key}")

    keys_to_delete = []
    for key in atrs.keys():
        symbol, timeframe = key.split("_")
        if symbol not in current_symbols or timeframe not in current_timeframes:
            keys_to_delete.append(key)

    for key in keys_to_delete:
        del atrs[key]
        logger.debug(f"Removed atr for {key}")

    return detectors, atrs



def fetch_support_resistance(symbol, liq_levels):

    # fetch support and resistance levels
    liq_data = liq_levels[symbol]  # Retrieve the list of values for 'symbol'

    # Assign the values to variables
    support_lower = min(liq_data)
    support_upper = sorted(liq_data)[1]  # Second lowest value

    resistance_upper = max(liq_data)
    resistance_lower = sorted(liq_data)[-2]  # Second highest value


    return support_upper, support_lower, resistance_upper, resistance_lower







def execute_strategies(client: DydxClient, detectors: dict, atrs: dict, liq_levels: defaultdict(list), first_iteration: bool, symbol: str, timeframe: str, df: pd.DataFrame, signals: list):

    current_time = datetime.datetime.now()
    minutes = current_time.minute
    timeframe_minutes = timeframe_to_minutes(timeframe)

    if not (minutes % timeframe_minutes):

        # If it's the first iteration, add all candles to the ATR. Otherwise, add only the last candle.
        if first_iteration:
            for _, candle in df.iterrows():
                atr = atrs[f"{symbol}_{timeframe}"]
                atr.add_input_value(candle)
                if not atr:
                    logger.debug(f"ATR not available for {symbol} on {timeframe} - no. input values: {len(atr.input_values)} - Needs: {atr.period}")
            last_closed_candle = df.iloc[-1]
        else:
            last_closed_candle = df.iloc[-1]
            atr = atrs[f"{symbol}_{timeframe}"]
            atr.add_input_value(last_closed_candle)
           
       

        # fetch support and resistance levels
        support_upper, support_lower, resistance_upper, resistance_lower = fetch_support_resistance(symbol, liq_levels)
        orders = list()
        for strategy_function_name in client.config['strategies'][0]['strategy_functions']:
            strategy_function = globals()[strategy_function_name]
            detector_key = f"{symbol}_{timeframe}_{strategy_function_name}"
            detector = detectors[detector_key]


            try:
                logger.debug(f"Executing strategy {strategy_function_name} for {symbol} on {timeframe}")
                if strategy_function_name == "doubleBottomEntry":
                    signal = strategy_function(last_closed_candle, detector, support_zone_upper=support_upper, support_zone_lower=support_lower)

                elif strategy_function_name == "doubleTopEntry":
                    signal = strategy_function(last_closed_candle, detector, ressist_zone_upper=resistance_upper, ressist_zone_lower=resistance_lower)
                
                elif strategy_function_name == "liqSweepEntry":
                    signal = strategy_function(last_closed_candle, detector, upper_liq_level=resistance_upper, lower_liq_level=support_lower )

            except Exception as e:
                logger.error(f"Error while executing strategy for {symbol} on {timeframe}: {e}")
           
            try:

                if signal[1] in ('SELL', 'BUY'):
                    logger.debug(f"Executing signal {strategy_function_name} for {symbol} on {timeframe} - side: {signal[1]}")

                    size = float(client.order_size(symbol, client.config['position_size']))

                    logger.info(f"\033[92mPlacing {signal[1]} order for {symbol} with size {size}\033[0m") 
                    order = client.place_market_order(symbol=symbol, size=size, side=signal[1], atr=atr[-1], trigger_candle=signal[0])
                    
                    signals.append((symbol, timeframe, signal[2], order))

                elif signal[1] is None:
                    logger.info(f"No signal for symbol: {symbol} on TF: {timeframe} - {strategy_function_name}")
                else:
                    logger.warning(f"Invalid signal: {signal[1]}")
                
            except Exception as e:
                logger.error(f"Error while executing signal for {symbol} on {timeframe}: {e} - {strategy_function_name}")




def execute_main(client: DydxClient, json_file_path: str, liq_levels: defaultdict(list), position_storage: PositionStorage):

    try:
        detectors, atrs = initialize_detectors(client)

        # Initialize the last hash as an empty string
        process_json_file.last_hash = ''

        # Initialize first_iteration as True
        first_iteration = True

        # tmp list for Liq. zones that needs to be updated:
        liq_zones_to_be_updated = list()
        

        while True:
            try:
                # Get the current time
                current_time = datetime.datetime.now()

                # Calculate the remaining seconds until the next minute
                remaining_seconds = 60 - current_time.second

                # Sleep for the remaining seconds
                time.sleep(remaining_seconds)

                res = process_json_file(json_file_path)

                # update active symbols & update entryStrat obj:
                if res is not None:
                    liq_levels = res
                    symbols_added = update_config_with_symbols(liq_levels, client)
                    if symbols_added:
                        detectors, atrs = initialize_detectors(client, detectors, atrs)
                        for sym in symbols_added:
                            msg = f"\033[92m{chr(0x2713)}\033[0m {sym} Active For Trading"
                            send_telegram_message(client.config['bot_token'], client.config['chat_ids'], msg, pass_time_limit=True)


                # create unique sets of all symbols and timeframes & fetch df for each:
                all_symbols = set(symbol for strategy in client.config['strategies'] for symbol in strategy['symbols'])
                all_timeframes = set(timeframe for strategy in client.config['strategies'] for timeframe in strategy['timeframes'])
                all_symbol_df = asyncio.run(get_all(all_symbols, all_timeframes, first_iteration))

                
                # execute strategy
                signals = list()
                for (symbol, timeframe), df in all_symbol_df.items():
                    execute_strategies(client, detectors, atrs, liq_levels, first_iteration, symbol, timeframe, df, signals)
                    #signals.append(res)
                logger.debug(f"All orders in each iteration is stored in signals: {signals}")


                # if signal: send TG msg. for open orders & save info to DB: ( skal stadig laves)
                if signals:
                    #msg = client.fetch_all_open_position(open_pos=sum([len(tup) for tup in signals]))
                    msg = client.fetch_all_open_position(open_pos=len(signals))
                    send_telegram_message(client.config['bot_token'], client.config['chat_ids'], msg, pass_time_limit=True)

                    for signal in signals:
                        symbol, tf, entry_strat_type, order = signal
                        if len(order) == 3:
                            res = next((pos for pos in client.client.private.get_positions(status='Open').data.get('positions') if pos['market'] == symbol), None)
                            if res:
                                position_storage.insert_position(res, entry_strat_type, tf)


                check_liquidation_zone(liq_levels, client, liq_zones_to_be_updated)
                
                # Cancels all orders for trading pairs which don't have an open position:
                client.purge_no_pos_orders()


                # Send msg in TG when orders are closed:
                close_msg = client.send_tg_msg_when_trade_closed()
                if close_msg:
                    for msg in close_msg:
                        send_telegram_message(client.config['bot_token'], client.config['chat_ids'], msg, pass_time_limit=True)


                first_iteration = False

                # Sleep for some time before executing the strategies again (e.g., 60 seconds)
                logger.debug("Sleeping untill next minute")

                # Sleep for 1 second to ensure it runs at the beginning of the minute
                time.sleep(1)


            except Exception as e:
                logger.error(f"An error occurred in the main execution loop: {e}")
                logger.exception(e)

    finally:
        try:
            position_storage.close()
        except Exception as e:
            logger.error(f"An error occurred while closing the position storage: {e}")
            logger.exception(e)


def main():
    logger.info("Initializing detectors")
    client = DydxClient()

    # Initialize PositionStorage
    position_storage = PositionStorage('positions.db')

    # Start the bot in a separate thread
    bot_thread = threading.Thread(target=execute_main, args=(client, '/opt/tvserver/database.json', defaultdict(list), position_storage))
    bot_thread.start()

    # Run the bot in the main thread
    bot_main(client.config['bot_token'], client)


if __name__ == '__main__':
    main()


