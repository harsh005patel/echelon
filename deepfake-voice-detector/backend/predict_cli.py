import argparse
import torch
import numpy as np
import os
import sys
from loguru import logger

# Add current directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.deepfake_detector import DeepfakeDetector
from models.audio_preprocessor import AudioPreprocessor
from models.feature_extractor import FeatureExtractor

def main():
    parser = argparse.ArgumentParser(description="Deepfake Voice Detector CLI")
    parser.add_argument("audio_path", help="Path to the audio file to analyze")
    parser.add_argument("--weights_dir", default="models/weights", help="Directory containing model weights")
    args = parser.parse_args()

    audio_path = args.audio_path
    if not os.path.exists(audio_path):
        logger.error(f"File not found: {audio_path}")
        return

    logger.info(f"Analyzing {audio_path}...")

    # Load components
    try:
        detector = DeepfakeDetector(weights_dir=args.weights_dir)
        preprocessor = AudioPreprocessor()
        extractor = FeatureExtractor()
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return

    # Process Audio
    try:
        # Load
        audio, sr = preprocessor.load_audio(audio_path)
        
        # Preprocess (simple normalization for now)
        audio = preprocessor.normalize_loudness(audio)
        
        # Ensure length (pad/trim to ~3s for consistency with training if needed)
        # The models have adaptive pooling/GRUs, but consistent input is better. 
        # Training loop uses 3s (16000 * 3).
        target_len = 16000 * 3
        if len(audio) > target_len:
             # Take center segment for inference
            start = (len(audio) - target_len) // 2
            audio = audio[start:start+target_len]
        else:
            padding = target_len - len(audio)
            audio = np.pad(audio, (0, padding))

        # Extract features
        features = extractor.extract_all(audio)

        # Predict
        result = detector.predict(features, audio)
        
        print("\n" + "="*40)
        print(f"File: {os.path.basename(audio_path)}")
        print(f"Prediction: {'FAKE' if result['is_deepfake'] else 'REAL'}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("-" * 40)
        print("Individual Model Scores (Fake Probability):")
        print(f"  CNN-LSTM: {result['individual_scores']['cnn_lstm']:.4f}")
        print(f"  ResNet-SE: {result['individual_scores']['resnet_se']:.4f}")
        print(f"  RawNet:    {result['individual_scores']['rawnet']:.4f}")
        print("="*40 + "\n")

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
