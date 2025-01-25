import asyncio
import websockets
import json
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import socket
from dotenv import load_dotenv
from speechmatics.client import WebsocketClient
from speechmatics.models import ConnectionSettings, TranscriptionConfig
import sounddevice as sd
import numpy as np

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def get_speechmatics_api_key():
    api_key = os.getenv("SPEECHMATICS_API_KEY")
    if not api_key:
        logger.error("SPEECHMATICS_API_KEY environment variable not found")
        raise ValueError("SPEECHMATICS_API_KEY not set")
    return api_key

class TranscriptionManager:
    def __init__(self, loop):
        self.ws_client = None
        self.stream = None
        self.websocket = None
        self.running = False
        self.loop = loop
        self._executor = ThreadPoolExecutor()

    async def send_message(self, message):
        """Send a message to the WebSocket client."""
        if self.websocket and self.websocket.open:
            try:
                await self.websocket.send(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message: {e}")

    async def start_transcription(self, websocket):
        """Start transcription."""
        if self.running:
            logger.warning("Transcription already running. Restarting...")
            await self.stop_transcription()

        self.websocket = websocket
        self.running = True

        settings = ConnectionSettings(
            url="wss://eu.rt.speechmatics.com/v2",
            auth_token=get_speechmatics_api_key()
        )
        
        config = TranscriptionConfig(
            language="en",
            enable_partials=True,
            max_delay=2,
            enable_entities=True
        )

        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio status: {status}")
            if self.running:
                audio_data = indata.flatten().tobytes()
                try:
                    self.ws_client.send_audio(audio_data)
                except Exception as e:
                    logger.error(f"Error sending audio: {e}")

        try:
            self.stream = sd.InputStream(
                channels=1,
                samplerate=16000,
                dtype=np.int16,
                callback=audio_callback,
                blocksize=1600  # 100ms chunks
            )
            
            self.ws_client = WebsocketClient(settings)
            
            def handle_transcript(msg):
                if msg["message"] == "AddTranscript":
                    text = msg["metadata"]["transcript"]
                    is_partial = msg["metadata"]["is_partial"]
                    asyncio.run_coroutine_threadsafe(
                        self.send_message({
                            "type": "partial" if is_partial else "final", 
                            "text": text
                        }), self.loop
                    )
                elif msg["message"] == "Error":
                    logger.error(f"Speechmatics error: {msg}")
                    asyncio.run_coroutine_threadsafe(
                        self.send_message({
                            "type": "error",
                            "text": msg.get("metadata", {}).get("reason", "Unknown error")
                        }), self.loop
                    )

            self.ws_client.add_event_handler(handle_transcript)
            
            logger.info("Starting Speechmatics client...")
            await self.loop.run_in_executor(
                self._executor,
                self.ws_client.run_synchronously,
                config
            )
            self.stream.start()
            logger.info("Transcription started successfully")

        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            await self.send_message({"type": "error", "text": str(e)})
            await self.stop_transcription()

    async def stop_transcription(self):
        """Stop transcription."""
        if not self.running:
            logger.info("No transcription running to stop.")
            return

        logger.info("Stopping transcription...")
        self.running = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if self.ws_client:
            self.ws_client.close()
            self.ws_client = None

        await self.send_message({"type": "status", "text": "Transcription stopped"})
        logger.info("Transcription fully stopped")

async def handle_websocket(websocket, path, loop):
    """Handle WebSocket connections."""
    logger.info("New WebSocket connection")
    manager = TranscriptionManager(loop)

    try:
        async for message in websocket:
            logger.info(f"Received WebSocket message: {message}")
            if message == "start":
                await manager.start_transcription(websocket)
            elif message == "stop":
                await manager.stop_transcription()
            else:
                logger.warning(f"Unknown command: {message}")
    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed")
    finally:
        await manager.stop_transcription()

def start_webserver(port, directory):
    """Start a simple web server to serve static files."""
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
    httpd = HTTPServer(("", port), Handler)
    logger.info(f"Web server running on http://localhost:{port}")
    httpd.serve_forever()

async def main():
    """Start the WebSocket server."""
    loop = asyncio.get_running_loop()

    host = "0.0.0.0" if os.getenv("RAILWAY_ENVIRONMENT") else "localhost"
    port = int(os.getenv("PORT", 8765))

    webserver_port = 8000
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", webserver_port))
                break
        except OSError:
            webserver_port += 1

    webserver_thread = threading.Thread(
        target=start_webserver,
        args=(webserver_port, os.path.dirname(os.path.abspath(__file__)))
    )
    webserver_thread.daemon = True
    webserver_thread.start()

    async with websockets.serve(
        lambda ws, path: handle_websocket(ws, path, loop),
        host,
        port,
    ):
        logger.info(f"WebSocket server running on ws://{host}:{port}")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
