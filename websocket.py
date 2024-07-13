import asyncio
import traceback

import websockets
import logging
import json


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class WebsocketHandler:
    def __init__(self, on_message):
        self.server = None
        self.clients = set()
        self.board = None
        self.done = False
        self.on_message = on_message
        self.shutdown_signal = asyncio.Event()

    async def handle_websocket(self, websocket, path):
        logger.info(f"WebSocket connection established with {path}")
        self.clients.add(websocket)
        try:
            async for message in websocket:
                logger.info(f"Message from {path}: {message}")
                await self.process_websocket_message(message)
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"WebSocket connection with {path} closed: {e}")
        except Exception as e:
            logger.error(f"Error in WebSocket connection with {path}: {e}")
        finally:
            self.clients.remove(websocket)
            logger.info(f"WebSocket connection with {path} terminated")

    async def start_websocket_server(self, port):
        logger.info(f"WebSocket server starting on port {port}")
        self.server = await websockets.serve(self.handle_websocket, "", port)
        await self.shutdown_signal.wait()
        await self.server.close()


    def stop(self):
        self.shutdown_signal.set()

    async def process_websocket_message(self, message):
        try:
            msg = json.loads(message)
            logger.info(f"Command received: {message}")
            await self.broadcast_websocket_message(json.dumps({
                'address': 'log',
                'message': f"Command '{message}' received"
            }))
            self.on_message(msg)
            await self.broadcast_websocket_message(json.dumps({
                'address': 'log',
                'status': 'success',
                'message': f"Command '{msg['command']}' processed"
            }))
        except Exception as error:
            logger.error(f'Error processing message: {error}')
            traceback.print_exc()
            await self.broadcast_websocket_message(json.dumps({
                'address': 'log',
                'status': 'error',
                'message': f"Command '{message}' failed",
                'error': str(error)
            }))

    async def broadcast_websocket_message(self, message):
        logger.info("Broadcasting message: " + message)
        for client in self.clients:
            await client.send(message)
