from dydx3.helpers.request_helpers import generate_now_iso
from dydx3.constants import WS_HOST_MAINNET
from connectors import dydx_connector
from dydx3 import Client
import websockets
import asyncio
import json



class RealTimePriceFetcher:
    def __init__(self, symbols):
        self.latest_prices = {}
        self.stop_signal = False
        self.symbols = symbols


    client = dydx_connector.get_client()

    now_iso_string = generate_now_iso()

    signature = client.private.sign(
        request_path='/ws/accounts',
        method='GET',
        iso_timestamp=now_iso_string,
        data={},
    )

    req = {
        'type': 'subscribe',
        'channel': 'v3_markets',
        'accountNumber': '0',
        'apiKey': client.api_key_credentials['key'],
        'passphrase': client.api_key_credentials['passphrase'],
        'timestamp': now_iso_string,
        'signature': signature,
    }

    async def _fetch_prices(self):
        try:
            async with websockets.connect(WS_HOST_MAINNET) as websocket:
                await websocket.send(json.dumps(self.req))

                while True:
                    try:    
                        res = await websocket.recv()
                        data = json.loads(res)

                        if 'contents' in data:
                            self.latest_prices = data['contents']
                    except (websockets.exceptions.ConnectionClosedError, json.JSONDecodeError) as e:
                        print(f'Error while receiving data: {e}')
                        break
        except websockets.exceptions.WebSocketException as e:
            print(f'Error while connecting to WebSocket: {e}')


    def start(self):
        self.stop_signal = False
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        asyncio.run(self._fetch_prices())
        
    def stop(self):
        self.stop_signal = True


    def get_latest_prices(self, symbol):
        if isinstance(self.latest_prices, dict):
            return self.latest_prices.get(symbol, {}).get('indexPrice', "websocket needs a few seconds to connect")

