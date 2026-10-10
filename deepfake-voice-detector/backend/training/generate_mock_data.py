import os
import numpy as np
import soundfile as sf
from scipy import signal
from loguru import logger

OUTPUT_DIR = "data/raw"
SAMPLE_RATE = 16000
DURATION = 3.0

def apply_formants(source, sr, formants, bandwidths):
    """Filter glottal source through vocal tract formants using resonator filters."""
    filtered = source.copy()
    for f, bw in zip(formants, bandwidths):
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * f / sr
        b = [1 - 2*r*np.cos(theta) + r**2]
        a = [1, -2*r*np.cos(theta), r**2]
        filtered = signal.lfilter(b, a, filtered)
    return filtered

def generate_human_voice(seed=0, duration=DURATION, sr=SAMPLE_RATE):
    """
    Generate realistic human speech acoustics:
    - Natural glottal pulse train with realistic F0 (100-240 Hz)
    - Natural micro-prosody (vibrato 4-6 Hz and pitch declination)
    - Multi-formant vocal tract resonances (F1-F4)
    - Syllable envelope modulation (2-4 Hz)
    - Natural unvoiced fricative/aspiration components
    """
    np.random.seed(seed)
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # 1. Base F0 with natural speech intonation curve
    base_f0 = 110 + (seed % 15) * 8  # 110 Hz to 230 Hz (covers male & female range)
    # Intonation: slight rise then fall across the utterance
    intonation = np.sin(np.pi * t / duration) * 15
    # Micro-prosodic vibrato/jitter (5 Hz tremor)
    jitter = np.sin(2 * np.pi * 5.2 * t) * 3 + np.random.normal(0, 0.8, n_samples)
    f0_track = np.clip(base_f0 + intonation + jitter, 80, 300)

    # 2. Phase accumulator for glottal source
    phase = 2 * np.pi * np.cumsum(f0_track) / sr

    # 3. Glottal source (rich harmonic train with natural rolloff)
    source = np.zeros(n_samples)
    for harmonic in range(1, 20):
        amplitude = 1.0 / (harmonic ** 1.3)
        source += amplitude * np.sin(harmonic * phase)

    # 4. Syllable amplitude envelope (cadence of speech: 3 syllables per second)
    syllable_rate = 2.8 + (seed % 4) * 0.4
    syllable_env = np.clip(0.6 * np.sin(2 * np.pi * syllable_rate * t) + 0.4, 0.05, 1.0)
    source *= syllable_env

    # 5. Formant frequencies (vocal tract filtering for vowels like /a/, /e/, /i/, /o/, /u/)
    vowel_sets = [
        ([730, 1090, 2440, 3500], [80, 90, 120, 150]),   # /a/
        ([270, 2290, 3010, 3600], [60, 100, 120, 150]),  # /i/
        ([530, 1840, 2480, 3500], [70, 90, 120, 150]),   # /e/
        ([300, 870, 2240, 3400], [65, 80, 110, 140]),    # /u/
        ([570, 840, 2410, 3500], [70, 80, 120, 150]),    # /o/
    ]
    formants, bandwidths = vowel_sets[seed % len(vowel_sets)]
    speech = apply_formants(source, sr, formants, bandwidths)

    # 6. Natural breath / aspiration noise
    breath = np.random.normal(0, 0.015, n_samples) * syllable_env
    speech += breath

    # 7. Normalize
    speech = speech / (np.max(np.abs(speech)) + 1e-8) * 0.7
    return speech.astype(np.float32)

def generate_deepfake_voice(seed=0, duration=DURATION, sr=SAMPLE_RATE):
    """
    Generate deepfake/synthesized speech with classic acoustic anomalies:
    - Robotic rigid pitch (flatlined F0, zero micro-prosody)
    - Neural vocoder phase smearing and metallic high-frequency buzz
    - Artificial high-frequency cutoff (brick-wall filter at 3.8-7 kHz)
    - Concatenation / frame boundary jumps
    - Mel-inversion periodic ripple
    """
    np.random.seed(1000 + seed)
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    artifact_type = seed % 4

    # 1. Rigid artificial pitch (dead flatline F0 common in synthetic speech)
    base_f0 = 130 + (seed % 10) * 12
    # Zero natural tremor or intonation - dead flat
    phase = 2 * np.pi * base_f0 * t

    # 2. Synthetic harmonic source
    source = np.zeros(n_samples)
    for h in range(1, 15):
        # Neural vocoder overtone imbalance
        amp = 1.0 / (h ** 0.8)
        source += amp * np.sin(h * phase)

    if artifact_type == 0:
        # Metallic high-frequency buzz (neural vocoder / HiFi-GAN phase ringing)
        buzz = 0.25 * np.sin(2 * np.pi * 3800 * t) + 0.15 * np.sin(2 * np.pi * 5600 * t)
        audio = source + buzz

    elif artifact_type == 1:
        # High-frequency brick-wall spectral cutoff (common in 8kHz/16kHz TTS models)
        nyq = sr / 2
        cutoff = 3800.0 / nyq
        b, a = signal.butter(6, cutoff, btype='low')
        audio = signal.lfilter(b, a, source)
        # Add robotic robotic phase noise
        audio += 0.05 * np.random.uniform(-1, 1, n_samples)

    elif artifact_type == 2:
        # Frame boundary discontinuities (neural frame jumps every 20ms = 320 samples)
        audio = source.copy()
        frame_len = 320
        for start in range(0, n_samples - frame_len, frame_len):
            if (start // frame_len) % 3 == 0:
                audio[start:start+16] = 0.0  # dropout
                audio[start+16:start+frame_len] *= 1.3  # gain jump

    else:
        # Griffin-Lim phase inversion artifact (harsh comb-filtering)
        comb_delay = int(sr * 0.005)  # 5ms comb filter
        delayed = np.pad(source[:-comb_delay], (comb_delay, 0))
        audio = source + 0.8 * delayed

    # 3. Add synthetic buzz envelope
    syllable_env = np.clip(0.5 * np.sin(2 * np.pi * 3.0 * t) + 0.5, 0.1, 1.0)
    audio *= syllable_env

    # 4. Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.7
    return audio.astype(np.float32)

def generate_mock_dataset(num_samples=30):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info(f"Generating realistic speech and deepfake dataset in {OUTPUT_DIR} ({num_samples} real, {num_samples} fake)...")

    # Generate Real audio files
    for i in range(num_samples):
        audio = generate_human_voice(seed=i)
        filename = os.path.join(OUTPUT_DIR, f"real_{i:03d}.wav")
        sf.write(filename, audio, SAMPLE_RATE)

    # Generate Fake audio files
    for i in range(num_samples):
        audio = generate_deepfake_voice(seed=i)
        filename = os.path.join(OUTPUT_DIR, f"fake_{i:03d}.wav")
        sf.write(filename, audio, SAMPLE_RATE)

    logger.info(f"Dataset generation complete. Created {num_samples * 2} audio files.")

if __name__ == "__main__":
    generate_mock_dataset(30)
