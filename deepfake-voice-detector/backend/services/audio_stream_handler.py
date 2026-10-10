import asyncio
from loguru import logger
import collections
from services.detection_service import get_detection_service
from utils.config import settings

class AudioStreamHandler:
    def __init__(self):
        self.detection_service = get_detection_service()
        self.buffer = bytearray()
        # Ensure we have enough data for at least one chunk (e.g., 3 seconds at 16kHz, 16-bit)
        # 3 * 16000 * 2 bytes = 96000 bytes
        self.chunk_size_bytes = int(settings.CHUNK_DURATION * settings.SAMPLE_RATE * 2) 
        self.overlap_bytes = int(settings.OVERLAP * settings.SAMPLE_RATE * 2)
        
    async def process_stream(self, data_chunk: bytes):
        """
        Receives small chunks from WS, buffers them, and triggers detection.
        Returns result if detection ran, else None.
        """
        self.buffer.extend(data_chunk)
        
        # Trigger detection every 1.5 seconds of audio (48000 bytes) for fast responsive UI
        min_process_bytes = int(1.5 * settings.SAMPLE_RATE * 2)
        
        if len(self.buffer) >= self.chunk_size_bytes:
            processing_data = self.buffer[:self.chunk_size_bytes]
            stride_bytes = self.chunk_size_bytes - self.overlap_bytes
            result = await self.detection_service.process_audio_chunk(bytes(processing_data))
            self.buffer = self.buffer[stride_bytes:]
            return result
        elif len(self.buffer) >= min_process_bytes and len(data_chunk) == 0:
            # Flushed or stream paused
            return await self.flush()
            
        return None

    async def flush(self):
        """
        Process whatever audio remains in the buffer when capture stops,
        even if shorter than 3 seconds (as long as at least 0.3s of audio exists).
        """
        min_bytes = int(0.3 * settings.SAMPLE_RATE * 2) # ~0.3 seconds minimum
        if len(self.buffer) >= min_bytes:
            data = bytes(self.buffer)
            self.buffer = bytearray()
            logger.info(f"Flushing remaining {len(data)} audio bytes on capture stop")
            return await self.detection_service.process_audio_chunk(data)
        self.buffer = bytearray()
        return None
    
    def reset(self):
        self.buffer = bytearray()
