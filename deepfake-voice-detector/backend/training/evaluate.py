import torch
from models.deepfake_detector import DeepfakeDetector
from training.dataset_loader import DeepfakeDataset
from torch.utils.data import DataLoader
from loguru import logger
import numpy as np

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load detector (loads weights automatically)
    detector = DeepfakeDetector(weights_dir="models/weights")
    
    # Use same dataset for simplicity in this demo, usually would use test set
    dataset = DeepfakeDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    correct = 0
    total = 0
    
    logger.info("Starting evaluation...")
    
    for batch in dataloader:
        try:
            # Reconstruct dictionary for predict()
            # Dataset returns tensors, predict expects numpy/list for features, raw for audio
            # But wait, predict expects: features dict, raw_audio
            
            features = {
                'mfcc': batch['mfcc'].squeeze(0).numpy(),
                'mel_spec': batch['mel_spec'].squeeze(0).numpy(),
                # Add placeholders if needed, but detector only uses these 2 + raw
                # Actually detector expects standard dict keys from extractor
            }
            raw_audio = batch['audio'].squeeze(0).numpy()
            label = batch['label'].item()
            
            result = detector.predict(features, raw_audio)
            
            start_msg = f"Sample {total+1}: True={label}, Pred={result['is_deepfake']} ({result['confidence']:.2f})"
            logger.info(start_msg)
            
            pred_label = 1 if result['is_deepfake'] else 0
            if pred_label == label:
                correct += 1
            total += 1
            
        except Exception as e:
            logger.error(f"Eval error: {e}")
            
    if total > 0:
        logger.info(f"Accuracy: {correct/total:.2%}")
    else:
        logger.warning("No samples evaluated.")

if __name__ == "__main__":
    evaluate()
