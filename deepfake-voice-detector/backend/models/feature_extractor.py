import numpy as np
import librosa
import scipy.signal
import torch
from loguru import logger
# import parselmouth # Uncomment if Praat integration is needed, handled as optional for now to avoid install issues if missing

class FeatureExtractor:
    def __init__(self, sample_rate=16000, n_mfcc=40, n_lfcc=20, n_mels=128):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_lfcc = n_lfcc
        self.n_mels = n_mels
        
    def extract_spectral_features(self, audio):
        """Extract MFCC, Mel-Spectrogram, and basic spectral stats."""
        # Mel-Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, n_mels=self.n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # MFCC
        mfcc = librosa.feature.mfcc(S=mel_spec_db, n_mfcc=self.n_mfcc)
        
        # Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        
        # Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        
        return {
            "mel_spec": mel_spec_db,
            "mfcc": mfcc,
            "centroid": centroid,
            "rolloff": rolloff,
            "zcr": zcr
        }

    def extract_prosodic_features(self, audio):
        """Extract Pitch (F0) and simple prosody."""
        # F0 using librosa's yin (simpler than praat for pure python deps)
        f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        
        # Clean F0 (remove NaNs)
        f0_clean = np.nan_to_num(f0)
        
        return {
            "f0": f0_clean,
            "jitter": 0.0, # Placeholder, requires more complex logic or parselmouth
            "shimmer": 0.0 # Placeholder
        }

    def extract_phase_features(self, audio):
        """Extract phase information."""
        S = librosa.stft(audio)
        phase = np.angle(S)
        
        # Group Delay (derivative of phase with respect to frequency)
        # unwrap phase
        unwrapped_phase = np.unwrap(phase, axis=0)
        group_delay = -np.diff(unwrapped_phase, axis=0)
        
        return {
            "phase": phase,
            "group_delay": group_delay
        }

    def extract_temporal_features(self, mfcc):
        """Extract Delta and Delta-Delta."""
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        return delta, delta2

    def extract_all(self, audio):
        """Aggregator."""
        spectral = self.extract_spectral_features(audio)
        prosodic = self.extract_prosodic_features(audio)
        phase = self.extract_phase_features(audio)
        delta, delta2 = self.extract_temporal_features(spectral['mfcc'])
        
        # Combine into a feature vector or dictionary appropriately for the model
        # For now, returning the dictionary
        features = {
            **spectral,
            **prosodic,
            **phase,
            "delta": delta,
            "delta2": delta2
        }
        return features
