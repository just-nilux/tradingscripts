import json
import datetime
import strategy
import threading


from time import sleep
from connectors import dydx_connector
from websocket_realtimePrices import RealTimePriceFetcher


with open('config.json') as config_file:
    config = json.load(config_file)


def main():

    # Initialize and start the RealTimePriceFetcher
    symbols = dydx_connector.get_all_symbols()
    price_fetcher = RealTimePriceFetcher(symbols)
    price_fetcher_thread = threading.Thread(target=price_fetcher.start)
    price_fetcher_thread.start()

    while True:
        minutes = datetime.datetime.now().minute

        for symbol in range(len(config['time_frame'])):
            for strat in range(len(config['time_frame'][symbol])):
                if not(minutes % config['DYDX_TIME_FRAMES'][config['time_frame'][symbol][strat]]):
                    df = strategy.strategy(dydx_connector.get_klines(config['symbols'][symbol], config['time_frame'][symbol][strat]), symbol, strat)

                    if df['LONG'][len(df) - 1]:
                        dydx_connector.place_order(config['symbols'][symbol], dydx_connector.order_size(config['symbols'][symbol], symbol, strat), 'BUY', 'MARKET', df)
                        
                        sleep(60)

        latest_price = price_fetcher.get_latest_prices(symbol=config['symbols'][symbol])
        print(latest_price)

        sleep(1)

if __name__ == '__main__':
    main()