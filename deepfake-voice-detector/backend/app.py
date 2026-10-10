from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import uvicorn
from contextlib import asynccontextmanager
import io

from utils.config import settings
# Import routers and services
from services.audio_stream_handler import AudioStreamHandler

# Configure logging
logger.add("logs/app.log", rotation="500 MB", level="INFO")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Deepfake Voice Detector...")
    # Pre-load models at startup
    from services.detection_service import get_detection_service
    service = get_detection_service()
    logger.info("Models loaded successfully.")
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

@app.post(f"{settings.API_V1_STR}/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """Upload an audio file for deepfake analysis."""
    from services.detection_service import get_detection_service
    service = get_detection_service()
    contents = await file.read()
    result = await service.process_audio_chunk(contents)
    return result

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
