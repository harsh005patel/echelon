import numpy as np
from loguru import logger
import time

from models.audio_preprocessor import AudioPreprocessor
from models.feature_extractor import FeatureExtractor
from models.deepfake_detector import DeepfakeDetector
from services.alert_service import AlertService

class DetectionService:
    def __init__(self):
        self.preprocessor = AudioPreprocessor()
        self.extractor = FeatureExtractor()
        self.detector = DeepfakeDetector()
        self.alert_service = AlertService()
        
    async def process_audio_chunk(self, audio_bytes):
        """
        Process a raw audio chunk (bytes), perform detection, and return results.
        Expected input: Raw PCM bytes (16-bit, 16kHz usually, but handled by preprocessor)
        """
        start_time = time.time()
        
        try:
            # 1. Convert bytes to numpy (assuming 16-bit PCM for now, or use load_audio logic)
            # This is a critical step for real-time. 
            # If incoming is raw bytes from websocket:
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # 2. Preprocess (VAD, Noise Reduce, Norm)
            # Note: For real-time stream, chunking usually happens before this or we process fixed windows.
            # Here we assume audio_bytes is already a chunk of appropriate size (e.g. 3s buffer)
            # But the preprocessor.preprocess does chunking.
            # Let's simplify: process the current buffer as one segment if possible.
            
            processed_chunks = self.preprocessor.preprocess(audio_data)
            
            results = []
            
            # If VAD removed everything, processed_chunks might be empty
            if len(processed_chunks) == 0:
                 return {"status": "silence", "confidence": 0.0}

            for chunk in processed_chunks:
                # 3. Extract Features
                features = self.extractor.extract_all(chunk)
                
                # 4. Predict
                prediction = self.detector.predict(features, chunk)
                results.append(prediction)
                
            # Aggregate results from chunks (max or avg)
            # Taking max confidence of being fake for safety
            if not results:
                return {"status": "processed", "is_deepfake": False, "confidence": 0.0}
                
            max_conf_idx = np.argmax([r['confidence'] for r in results])
            final_result = results[max_conf_idx]
            
            process_time = time.time() - start_time
            final_result['process_time'] = process_time
            
            # 5. Alert if needed
            if final_result['is_deepfake']:
                await self.alert_service.send_alert(final_result)
                
            return final_result
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return {"error": str(e)}
