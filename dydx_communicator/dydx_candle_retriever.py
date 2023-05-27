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
    timeframe: str = '1h',
    first_iteration: bool = False
) -> pd.DataFrame:

    async with sem:
        try:
            logger.info(f"Retrieving {symbol}")

            # Set the limit parameter based on whether it's the first iteration or not
            limit = 100 if first_iteration else 2

            candle_endpoint = f"{BASE_URL}/v3/candles/{symbol}"
            resp = await session.request('GET', url=candle_endpoint, params={'resolution': timeframe, 'limit': limit})
            response_json = await resp.json()
            candles = response_json["candles"]

            # Reverse the list of candles if it's the first iteration (API returns candles in descending order)
            if first_iteration:
                candles = candles[::-1]

            # Process all candles during the first iteration, and only the last closed candle in subsequent iterations
            processed_candles = []
            for candle in candles:
                candle_start = datetime.strptime(candle["startedAt"], "%Y-%m-%dT%H:%M:%S.%f%z")
                candle_end = candle_start + timedelta(seconds=durations[timeframe])

                now = datetime.now(tz=candle_end.tzinfo)

                # If it's not the first iteration and the current candle is not closed, break the loop
                if not first_iteration and now <= candle_end:
                    break

                # Create a pandas Series from the candle
                formatted_timestamp = datetime.strptime(candle['updatedAt'], "%Y-%m-%dT%H:%M:%S.%f%z").strftime("%Y-%m-%d %H:%M:%S")
                data = {'open': float(candle['open']), 'high': float(candle['high']), 'low': float(candle['low']), 'close': float(candle['close']), 'volume': float(candle['usdVolume'])}
                processed_candle = pd.Series(data, name=formatted_timestamp)
                processed_candles.append(processed_candle)

            return symbol, pd.DataFrame(processed_candles)

        except Exception as e:
            logger.error(f"Error occurred while retrieving {symbol}: {str(e)}")
            return symbol, None



async def get_all(symbols: List[str], timeframes: List[str], first_iteration: bool) -> Dict[str, Dict[str, Any]]:
    """
    Retrieves the latest closed candles for all symbols and timeframes.
    """
    sem = asyncio.Semaphore(CONCURRENT_LIMIT)
    async with aiohttp.ClientSession() as session:
        tasks = {get_klines_async(sem, session, symbol, timeframe=timeframe, first_iteration=first_iteration): (symbol, timeframe) 
                 for symbol in symbols
                 for timeframe in timeframes}
        results = await asyncio.gather(*tasks.keys(), return_exceptions=True)

        candles = {tasks[task]: result[1] for task, result in zip(tasks.keys(), results) if result is not None}

        return candles
