import asyncio
import aiohttp
import sys
from datetime import datetime, timedelta
from dydx3 import Client
from web3 import Web3

if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class CandleRetriever:
    DURATIONS = {
        '1MIN': 60,
        '5MINS': 5 * 60,
        '30MINS': 30 * 60,
        '15MINS': 15 * 60,
        '1HOUR': 60 * 60,
        '4HOURS': 60 * 60 * 4,
        '1DAY': 60 * 60 * 24
    }
    
    BASE_URL = "https://api.dydx.exchange"
    CONCURRENT_LIMIT = 20

    def __init__(self, symbols, limit, timeframe):
        self.symbols = symbols
        self.limit = limit
        self.timeframe = timeframe
        self.semaphore = asyncio.Semaphore(self.CONCURRENT_LIMIT)

    async def get_klines_async(self, session, symbol):
        async with self.semaphore, session.get(
                f"{self.BASE_URL}/v3/candles/{symbol}",
                params={'resolution': self.timeframe, 'limit': self.limit + 1}
        ) as resp:
            response_json = await resp.json()
            candle = self.find_latest_closed_candle(response_json["candles"])
            return symbol, candle

    @staticmethod
    def find_latest_closed_candle(candles):
        latest_closed_candle = None
        latest_closed_candle_start = None
        for candle in candles:
            candle_start = datetime.strptime(candle["startedAt"], "%Y-%m-%dT%H:%M:%S.%f%z")
            candle_end = candle_start + timedelta(seconds=CandleRetriever.DURATIONS[timeframe])

            now = datetime.now(tz=candle_end.tzinfo)
            if now >= candle_end and (latest_closed_candle_start is None or latest_closed_candle_start < candle_start):
                latest_closed_candle = candle
                latest_closed_candle_start = candle_start

        return latest_closed_candle

    async def get_all(self):
        async with aiohttp.ClientSession(trust_env=True) as session:
            tasks = [self.get_klines_async(session, symbol) for symbol in self.symbols]
            return await asyncio.gather(*tasks, return_exceptions=True)
