from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import uvicorn
from contextlib import asynccontextmanager

from utils.config import settings
# Import routers and services
from services.audio_stream_handler import AudioStreamHandler

# Configure logging
logger.add("logs/app.log", rotation="500 MB", level="INFO")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models, connect to DB, etc.
    logger.info("Starting up Deepfake Voice Detector...")
    # Initialize global services if needed to persist models in memory
    # For now, services instantiate models on init, which is fine (models are cached or lightweight enough? No, models should be singleton)
    # The detector loads weights on init. It might be better to load once.
    # But for now, we'll let each connection create a handler (which uses services).
    # TODO: Refactor for singleton DetectionService if memory usage is an issue.
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    stream_handler = AudioStreamHandler()
    
    try:
        while True:
            data = await websocket.receive_bytes()
            # Process audio stream
            result = await stream_handler.process_stream(data)
            
            if result:
                await websocket.send_json(result)
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
