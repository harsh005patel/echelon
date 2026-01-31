import os
import glob
import torch
from torch.utils.data import Dataset
from models.audio_preprocessor import AudioPreprocessor
from models.feature_extractor import FeatureExtractor
from loguru import logger
import numpy as np

class DeepfakeDataset(Dataset):
    def __init__(self, data_dir="data/raw", transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.preprocessor = AudioPreprocessor()
        self.extractor = FeatureExtractor()
        
        # Simple file collection based on naming convention
        self.real_files = glob.glob(os.path.join(data_dir, "real_*.wav"))
        self.fake_files = glob.glob(os.path.join(data_dir, "fake_*.wav"))
        self.all_files = self.real_files + self.fake_files
        self.labels = [0] * len(self.real_files) + [1] * len(self.fake_files)
        
        logger.info(f"Found {len(self.real_files)} real and {len(self.fake_files)} fake samples.")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        label = self.labels[idx]
        
        try:
            # 1. Load and Preprocess
            # Note: In a real training pipeline, we might process offline to save time.
            # Here we do on-the-fly for simplicity.
            audio, sr = self.preprocessor.load_audio(filepath)
            
            # Simple preprocess: Normalize only, assume VAD done or not needed for stock data
            audio = self.preprocessor.normalize_loudness(audio)
            
            # Ensure fixed length for batching (e.g. 3 seconds)
            target_len = 16000 * 3
            if len(audio) > target_len:
                audio = audio[:target_len]
            else:
                padding = target_len - len(audio)
                audio = np.pad(audio, (0, padding))
                
            # 2. Extract Features
            features = self.extractor.extract_all(audio)
            
            # Return dict compatible with our model arguments
            # Convert to tensors
            return {
                "audio": torch.FloatTensor(audio),
                "mfcc": torch.FloatTensor(features['mfcc']),
                "mel_spec": torch.FloatTensor(features['mel_spec']),
                "label": torch.LongTensor([label])
            }
            
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            # Return dummy valid data to avoid crashing loader
            return self.__getitem__((idx + 1) % len(self))
