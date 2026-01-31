import asyncio
from loguru import logger
import collections
from services.detection_service import DetectionService
from utils.config import settings

class AudioStreamHandler:
    def __init__(self):
        self.detection_service = DetectionService()
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
        
        if len(self.buffer) >= self.chunk_size_bytes:
            # Extract the chunk for processing
            processing_data = self.buffer[:self.chunk_size_bytes]
            
            # actually rolling window: remove stride amount
            stride_bytes = self.chunk_size_bytes - self.overlap_bytes
            
            # Send to detection
            result = await self.detection_service.process_audio_chunk(processing_data)
            
            # Update buffer: remove the stride
            self.buffer = self.buffer[stride_bytes:]
            
            return result
            
        return None
    
    def reset(self):
        self.buffer = bytearray()
