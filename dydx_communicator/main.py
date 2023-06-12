from set_strategy_entry_obj import doubleTopEntry, doubleBottomEntry, liqSweepEntry
from strategies.doubleBottomEntry import DoubleBottomDetector
from strategies.doubleTopEntry import DoubleTopDetector
from strategies.liqSweepEntry import SweepDetector
from object_handler import initialize_detectors, object_cleanup
from position_storage import PositionStorage
from collections import defaultdict
from dydxClient import DydxClient
from json_file_processor import process_json_file
from send_telegram_message import bot_main, send_telegram_message
from logger_setup import setup_logger
from dydx_candle_retriever import get_all
from datetime import datetime


import asyncio
import pandas_ta as ta
import pandas as pd
import threading
import json
import time

logger = setup_logger(__name__)



def send_update(client, symbol, updated):
    """
    Sends a telegram message about the update status of liquidity zones.

    Args:
    client (obj): Client object to connect with the server and get market data.
    symbol (str): The symbol for which the liquidity zones are being updated.
    updated (bool): Whether the liquidity zones have been updated.
    """
    msg = f"Liquidity Zones have been updated for {symbol}" if updated else f'Update Liquidity Zones: {symbol}'
    send_telegram_message(msg, pass_time_limit=updated)
    if not updated:
        logger.debug(msg)


def check_liquidation_zone(data: dict, client: DydxClient, liq_zones_to_be_updated: set, updated_liq_levels):
    """
    Check the liquidation zone for each symbol in the data and send a 
    Telegram message if the current price is outside the provided zone.

    Args:
    data (dict): A dictionary with symbols as keys and list of prices as values.
    client (obj): Client object to connect with the server and get market data.
    liq_zones_to_be_updated (set): Set of symbols whose liquidity zones need to be updated.
    """
    for symbol, prices in data.items():
        if len(prices) != 4:
            continue

        current_price = float(client.client.public.get_markets(market=symbol).data['markets'][symbol]['oraclePrice'])

        min_price, max_price = min(prices), max(prices)

        if symbol in liq_zones_to_be_updated and min_price < current_price < max_price and updated_liq_levels:
            send_update(client, symbol, True)
            liq_zones_to_be_updated.remove(symbol)

        elif not min_price < current_price < max_price:
            liq_zones_to_be_updated.add(symbol)
            send_update(client, symbol, False)




def update_config_with_symbols(data: defaultdict, client: DydxClient):
    """
    Update the 'symbols' list in each strategy in the client's config with unique symbols from the data dictionary. 
    Symbols are selected if their corresponding list in the data has exactly 4 elements. 
    The updated config is written back to the client's config.json file.

    Parameters:
    data (defaultdict): A dictionary with symbols as keys and lists as values.
    client (DydxClient): A client object with a 'config.json'.

    Returns:
    added_symbols (list): A list of new symbols that have been added to the strategies.
    """

    added_symbols = []
    deactivated_sym = []

    # Find symbols with length == 4 in defaultdict
    new_symbols = [k for k, v in data.items() if len(v) == 4]

    # Update the symbols in strategies 
    for strategy in client.config['strategies']:
        if set(new_symbols) != set(strategy['symbols']):
            added_symbols += list(set(new_symbols).difference(strategy['symbols']))  # Compute and store the difference
            deactivated_sym += list(set(strategy['symbols']).difference(new_symbols))
            strategy['symbols'] = list(set(new_symbols))  # Convert to set and back to list to remove duplicates


    # Write back the updated json to file
    if added_symbols:
        with open("config.json", 'w') as json_file:
            json.dump(client.config, json_file, indent=2)

    return added_symbols, deactivated_sym



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

    minutes = datetime.now().minute
    timeframe_minutes = timeframe_to_minutes(timeframe)

    if not (minutes % timeframe_minutes):

        last_closed_candle = df.iloc[-1]

        # If it's the first iteration, add all candles to the ATR. Otherwise, add only the last candle.
        atr = atrs[f"{symbol}_{timeframe}"]
        if first_iteration and not atr:
            for _, candle in df.iterrows():
                atr.add_input_value(candle)
                if not atr:
                    logger.debug(f"ATR not available for {symbol} on {timeframe} - no. input values: {len(atr.input_values)} - Needs: {atr.period}")
        else:
            atr.add_input_value(last_closed_candle)
           
       
        # fetch support and resistance levels
        support_upper, support_lower, resistance_upper, resistance_lower = fetch_support_resistance(symbol, liq_levels)

        for strategy in client.config['strategies']:
            for strategy_function_name in strategy['strategy_functions']:
                
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

                        size = float(client.order_size(symbol, client.config['position_size'], in_testmode=strategy['in_testmode']))

                        logger.info(f"\033[92mPlacing {signal[1]} order for {symbol} with size {size}\033[0m") 
                        order = client.place_market_order(symbol=symbol, size=size, side=signal[1], atr=atr[-1], trigger_candle=signal[0])
                        
                        signals.append((symbol, timeframe, signal[2], order))

                    elif signal[1] is None:
                        logger.info(f"No signal for symbol: {symbol} on TF: {timeframe} - {strategy_function_name}")
                    else:
                        logger.warning(f"Invalid signal: {signal[1]}")
                    
                except Exception as e:
                    logger.error(f"Error while executing signal for {symbol} on {timeframe}: {e} - {strategy_function_name}")




def execute_main(client: DydxClient, json_file_path: str, position_storage: PositionStorage):

    try:
        logger.info("Initializing detectors")
        detectors, atrs = initialize_detectors(client)

        # Initialize the last hash as an empty string
        process_json_file.last_hash = ''

        # Initialize liq_levels as empty dict:
        liq_levels = defaultdict(list)

        # Initialize first_iteration as True
        first_iteration = True

        #list for Liq. zones that needs to be updated:
        liq_zones_to_be_updated = set()
        

        while True:
            try:
                # Sleep for the remaining seconds untill the next minute:
                time.sleep(60-datetime.now().second)

                updated_liq_levels, updated_since_last_time = process_json_file(json_file_path)

                # update active symbols & update entryStrat obj:
                if updated_liq_levels and updated_since_last_time:
                    liq_levels = updated_liq_levels

                    symbols_added, deactivated_sym = update_config_with_symbols(liq_levels, client)
                    if symbols_added:
                        first_iteration = True
                        detectors, atrs = initialize_detectors(client, detectors, atrs)
                        for sym in symbols_added:
                            msg = f"{sym} Activated For Trading"
                            send_telegram_message(msg, pass_time_limit=True)
                    if deactivated_sym:
                        object_cleanup(client, detectors, atrs)
                        for sym in deactivated_sym:
                            msg = f"{sym} Deactivated For Trading"
                            send_telegram_message(msg, pass_time_limit=True)


                # create unique sets of all symbols and timeframes & fetch df for each:
                all_symbols = set(symbol for strategy in client.config['strategies'] for symbol in strategy['symbols'])
                all_timeframes = set(timeframe for strategy in client.config['strategies'] for timeframe in strategy['timeframes'])
                all_symbol_df = asyncio.run(get_all(all_symbols, all_timeframes, first_iteration))

                
                # execute strategy
                signals = list()
                for (symbol, timeframe), df in all_symbol_df.items():
                    execute_strategies(client, detectors, atrs, liq_levels, first_iteration, symbol, timeframe, df, signals)
                logger.debug(f"All orders in each iteration is stored in signals: {signals}")


                # if signal: send TG msg. for open orders & save info to DB: ( skal stadig laves)
                if signals:
                    msg = client.fetch_all_open_position(open_pos=len(signals))
                    send_telegram_message(msg, pass_time_limit=True)

                    for signal in signals:
                        symbol, tf, entry_strat_type, order = signal
                        if isinstance(order, list) and len(order) == 3:
                            res = next((pos for pos in client.client.private.get_positions(status='Open').data.get('positions') if pos['market'] == symbol), None)
                            if res:
                                position_storage.insert_position(res, entry_strat_type, tf)


                check_liquidation_zone(liq_levels, client, liq_zones_to_be_updated, updated_liq_levels)
                
                # Cancels all orders for trading pairs which don't have an open position:
                client.purge_no_pos_orders()


                # Send msg in TG when orders are closed:
                close_msg = client.send_tg_msg_when_trade_closed()
                if close_msg:
                    for msg in close_msg:
                        send_telegram_message(msg, pass_time_limit=True)


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
    client = DydxClient()
    send_telegram_message("Starting Algo Bot", pass_time_limit=True)



    # Initialize PositionStorage
    position_storage = PositionStorage('positions.db')

    # Start the bot in a separate thread
    bot_thread = threading.Thread(target=execute_main, args=(client, '/opt/tvserver/database.json', position_storage))
    bot_thread.start()

    # Run the bot in the main thread
    bot_main(client.config['bot_token'], client)


if __name__ == '__main__':
    main()