import pandas as pd
import asyncio
import aiohttp
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List
from logger_setup import setup_logger


# https://github.com/encode/httpx/issues/914
if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from dydx3 import Client
from web3 import Web3

durations = {
    '1MIN': 60,
    '5MINS': 5 * 60,
    '30MINS': 30 * 60,
    '15MINS': 15 * 60,
    '1HOUR': 60 * 60,
    '4HOURS': 60 * 60 * 4,
    '1DAY': 60 * 60 * 24
}

CONCURRENT_LIMIT = 20  # run at most this amount of requests at the same time
BASE_URL = "https://api.dydx.exchange"
logger = setup_logger(__name__)


async def get_klines_async(
    sem: asyncio.locks.Semaphore,
    session: aiohttp.ClientSession,
    symbol: str,
    timeframe: str = '1h'
) -> pd.Series:

    async with sem:
        try:
            logger.info(f"Retrieving {symbol}")

            # https://api.dydx.exchange/v3/candles/BTC-USD?limit=5&resolution=1MIN
            candle_endpoint = f"{BASE_URL}/v3/candles/{symbol}"

            resp = await session.request('GET', url=candle_endpoint, params={'resolution': timeframe, 'limit': 2})
            response_json = await resp.json()
            candles = response_json["candles"]

            # find latest closed candle
            latest_closed_candle = None
            latest_closed_candle_start = None
            for candle in candles:
                candle_start = datetime.strptime(candle["startedAt"], "%Y-%m-%dT%H:%M:%S.%f%z")
                candle_end = candle_start + timedelta(seconds=durations[timeframe])
                print('HOLA')
                now = datetime.now(tz=candle_end.tzinfo)

                if now > candle_end and (latest_closed_candle is None or candle_start > latest_closed_candle_start):
                    latest_closed_candle = candle
                    latest_closed_candle_start = candle_start
            
            # Create a pandas Series from the latest closed candle
            formatted_timestamp = datetime.strptime(latest_closed_candle['startedAt'], "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d %H:%M:%S")
            data = {'Open': float(latest_closed_candle['open']), 'High': float(latest_closed_candle['high']), 'Low': float(latest_closed_candle['low']), 'Close': float(latest_closed_candle['close']), 'Volume': float(latest_closed_candle['usdVolume'])}
            latest_closed_candle = pd.Series(data, name=formatted_timestamp)

            return symbol, latest_closed_candle

        except Exception as e:
            logger.error(f"Error occurred while retrieving {symbol}: {str(e)}")
            return symbol, None



async def get_all(symbols: List[str], timeframes: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Retrieves the latest closed candles for all symbols and timeframes.
    """
    sem = asyncio.Semaphore(CONCURRENT_LIMIT)
    async with aiohttp.ClientSession() as session:
        tasks = {get_klines_async(sem, session, symbol, timeframe=timeframe): (symbol, timeframe) 
                 for symbol in symbols
                 for timeframe in timeframes}
        results = await asyncio.gather(*tasks.keys(), return_exceptions=True)

        #candles = {tasks[task]: result for task, result in zip(tasks.keys(), results) if result is not None}
        candles = {tasks[task]: result[1] for task, result in zip(tasks.keys(), results) if result is not None}

        return candles


#async def get_all(symbols: List[str], timeframe: str = '1h') -> Dict[str, Dict[str, Any]]:
#    """
#    Retrieves the latest closed candles for all symbols.
#    """
#    sem = asyncio.Semaphore(CONCURRENT_LIMIT)
#    async with aiohttp.ClientSession() as session:
#        tasks = {get_klines_async(sem, session, symbol, timeframe=timeframe): symbol for symbol in symbols}
#        results = await asyncio.gather(*tasks.keys(), return_exceptions=True)
#
#        candles = {tasks[task]: result for task, result in zip(tasks.keys(), results) if result is not None}
#        return candles
