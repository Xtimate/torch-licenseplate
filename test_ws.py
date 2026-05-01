import asyncio
import json

import websockets


async def test():
    async with websockets.connect("ws://localhost:8000/webcam") as ws:
        with open("/home/xtimate/Downloads/licenseplat.jpg", "rb") as f:
            await ws.send(f.read())
        result = await ws.recv()
        print(json.loads(result))


asyncio.run(test())
