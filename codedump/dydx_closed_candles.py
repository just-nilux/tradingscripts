import requests
import asyncio
import aiohttp 
import sys

# https://github.com/encode/httpx/issues/914
if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from dydx3 import Client
from web3 import Web3

from datetime import datetime, timedelta

durations = {
    '1MIN': 60,
    '5MINS': 5 * 60,
    '30MINS': 30 * 60,
    '15MINS': 15 * 60,
    '1HOUR': 60 * 60,
    '4HOURS': 60 * 60 * 4,
    '1DAY': 60 * 60 * 24
}

CONCURRENT_LIMIT = 20 # run at most this amount of requests at the same time
BASE_URL = "https://api.dydx.exchange"

async def get_klines_async(
    sem: asyncio.locks.Semaphore,
    session: aiohttp.ClientSession,
    symbol: str, 
    limit: int = 50, 
    timeframe: str = '1h'
) -> dict:

    async with sem:
        try:
            print(f"Retrieving {symbol}")

            # add additional candle to query - because open candle is included
            limit += 1 

            # https://api.dydx.exchange/v3/candles/BTC-USD?limit=5&resolution=1MIN
            candle_endpoint = f"{BASE_URL}/v3/candles/{symbol}"

            resp = await session.request('GET', url=candle_endpoint, params={'resolution': timeframe, 'limit': limit})
            reponse_json = await resp.json()
            candles = reponse_json["candles"]

            # find latest closed candle
            latest_closed_candle = None
            latest_closed_candle_start = None
            for candle in candles:
                candle_start = datetime.strptime(candle["startedAt"], "%Y-%m-%dT%H:%M:%S.%f%z")
                candle_end = candle_start + timedelta(seconds=durations[timeframe])

                now = datetime.now(tz=candle_end.tzinfo)
                is_closed = now >= candle_end

                if is_closed and (latest_closed_candle is None or (latest_closed_candle_start is not None and latest_closed_candle_start < candle_start)):
                    latest_closed_candle = candle
                    latest_closed_candle_start = candle_start

            print(f"Done retrieving {symbol}")

            return (symbol, latest_closed_candle)
        except Exception as e:
            raise e

async def get_all(symbols, limit, timeframe):
    async with aiohttp.ClientSession(trust_env=True) as session:
        sem = asyncio.Semaphore(CONCURRENT_LIMIT)

        tasks = []
        for symbol in symbols:
            tasks.append(asyncio.ensure_future(get_klines_async(sem, session, symbol, limit, timeframe)))
        
        # gather all results - blocking..
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

if __name__ == '__main__':

    timeframe = "1h"

    symbols = ["BTC-USD", "ETH-USD", "ADA-USD", "LTC-USD"]

    symbolresults = asyncio.run(get_all(symbols, 2, '1MIN'))

    for symbol, candle in symbolresults:
        print(symbol, candle)


    # Little test - that failed..
    # client = Client(host="https://api.dydx.exchange")

    # markets = client.public.get_markets().data
    # print(markets)

    # candles = client.public.get_candles(
    #     market=["BTC-USD", "ETH-USD"],
    #     resolution='1DAY',
    #     limit=2
    # ).data
    # print(candles)

    # test code
    # example_response = {'candles': [{'startedAt': '2023-05-24T00:00:00.000Z', 'updatedAt': '2023-05-24T07:42:59.309Z', 'market': 'BTC-USD', 'resolution': '1DAY', 'low': '26593', 'high': '27229', 'open': '27227', 'close': '26692', 'baseTokenVolume': '2975.742', 'trades': '11876', 'usdVolume': '79878936.8186', 'startingOpenInterest': '2784.9792'}, {'startedAt': '2023-05-23T00:00:00.000Z', 'updatedAt': '2023-05-23T23:59:57.410Z', 'market': 'BTC-USD', 'resolution': '1DAY', 'low': '26807', 'high': '27500', 'open': '26858', 'close': '27227', 'baseTokenVolume': '6782.7847', 'trades': '25743', 'usdVolume': '184841278.7747', 'startingOpenInterest': '2665.0640'}]}
    # example_response_1min = {"candles":[{"startedAt":"2023-05-24T08:16:00.000Z","updatedAt":"2023-05-24T08:16:00.000Z","market":"BTC-USD","resolution":"1MIN","low":"26709","high":"26709","open":"26709","close":"26709","baseTokenVolume":"0","trades":"0","usdVolume":"0","startingOpenInterest":"2792.9046"},{"startedAt":"2023-05-24T08:15:00.000Z","updatedAt":"2023-05-24T08:15:37.000Z","market":"BTC-USD","resolution":"1MIN","low":"26709","high":"26709","open":"26709","close":"26709","baseTokenVolume":"13.5316","trades":"17","usdVolume":"361415.5044","startingOpenInterest":"2793.2357"},{"startedAt":"2023-05-24T08:14:00.000Z","updatedAt":"2023-05-24T08:14:58.297Z","market":"BTC-USD","resolution":"1MIN","low":"26707","high":"26709","open":"26707","close":"26709","baseTokenVolume":"0.11","trades":"4","usdVolume":"2937.822","startingOpenInterest":"2794.5074"},{"startedAt":"2023-05-24T08:13:00.000Z","updatedAt":"2023-05-24T08:13:18.742Z","market":"BTC-USD","resolution":"1MIN","low":"26700","high":"26702","open":"26700","close":"26702","baseTokenVolume":"0.6099","trades":"4","usdVolume":"16285.4278","startingOpenInterest":"2794.5474"},{"startedAt":"2023-05-24T08:12:00.000Z","updatedAt":"2023-05-24T08:12:48.015Z","market":"BTC-USD","resolution":"1MIN","low":"26690","high":"26701","open":"26700","close":"26701","baseTokenVolume":"2.7951","trades":"12","usdVolume":"74614.7724","startingOpenInterest":"2793.9810"}]}
    # example_response_1min = {"candles":[{"startedAt":"2023-05-24T08:18:00.000Z","updatedAt":"2023-05-24T08:18:00.000Z","market":"BTC-USD","resolution":"1MIN","low":"26710","high":"26710","open":"26710","close":"26710","baseTokenVolume":"0","trades":"0","usdVolume":"0","startingOpenInterest":"2793.8994"},{"startedAt":"2023-05-24T08:17:00.000Z","updatedAt":"2023-05-24T08:17:41.186Z","market":"BTC-USD","resolution":"1MIN","low":"26709","high":"26710","open":"26709","close":"26710","baseTokenVolume":"0.2308","trades":"11","usdVolume":"6164.4896","startingOpenInterest":"2793.8070"},{"startedAt":"2023-05-24T08:16:00.000Z","updatedAt":"2023-05-24T08:16:58.437Z","market":"BTC-USD","resolution":"1MIN","low":"26709","high":"26710","open":"26710","close":"26709","baseTokenVolume":"2.4263","trades":"7","usdVolume":"64804.8461","startingOpenInterest":"2792.9046"},{"startedAt":"2023-05-24T08:15:00.000Z","updatedAt":"2023-05-24T08:15:37.000Z","market":"BTC-USD","resolution":"1MIN","low":"26709","high":"26709","open":"26709","close":"26709","baseTokenVolume":"13.5316","trades":"17","usdVolume":"361415.5044","startingOpenInterest":"2793.2357"},{"startedAt":"2023-05-24T08:14:00.000Z","updatedAt":"2023-05-24T08:14:58.297Z","market":"BTC-USD","resolution":"1MIN","low":"26707","high":"26709","open":"26707","close":"26709","baseTokenVolume":"0.11","trades":"4","usdVolume":"2937.822","startingOpenInterest":"2794.5074"}]}
    
    # for candle in example_response_1min["candles"]:
    #     print(candle)

    #     startms = datetime.strptime(candle["startedAt"], "%Y-%m-%dT%H:%M:%S.%f%z")
    #     print(startms)

    #     endms = startms + timedelta(seconds=durations["1MIN"])
    #     print(endms)

    #     nowms = datetime.now(tz=endms.tzinfo)
    #     is_open = nowms < endms

    #     print("OPEN" if is_open else "CLOSED")
