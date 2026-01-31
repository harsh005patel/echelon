import os
import numpy as np
import soundfile as sf
from loguru import logger

OUTPUT_DIR = "data/raw"

def generate_tone(freq=440, duration=3, sample_rate=16000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Generate a simple tone
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    # Add some noise
    noise = np.random.normal(0, 0.05, audio.shape)
    return audio + noise

def generate_mock_dataset():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    logger.info(f"Generating mock dataset in {OUTPUT_DIR}...")
    
    # Generate 5 "Real" files (Clean tones)
    for i in range(5):
        # Vary frequency slightly
        freq = 440 + (i * 50) 
        audio = generate_tone(freq=freq)
        filename = os.path.join(OUTPUT_DIR, f"real_{i}.wav")
        sf.write(filename, audio, 16000)
        logger.info(f"Created {filename}")

    # Generate 5 "Fake" files (Distorted/Different pattern)
    for i in range(5):
        freq = 880 + (i * 50)
        audio = generate_tone(freq=freq)
        # Add synthetic artifacts (e.g., zeros every N samples)
        audio[::100] = 0
        # Add more noise
        audio += np.random.normal(0, 0.1, audio.shape)
        
        filename = os.path.join(OUTPUT_DIR, f"fake_{i}.wav")
        sf.write(filename, audio, 16000)
        logger.info(f"Created {filename}")
        
    logger.info("Mock dataset generation complete.")

if __name__ == "__main__":
    generate_mock_dataset()
