import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger
import os

# --- Model 1: CNN-LSTM ---
class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, num_layers=2, num_classes=2):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Calculate size after convolutions. Assumes input roughly (batch, 1, 40, ~300)
        # Using adaptive pool to force a fixed size for LSTM input
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 20)) # (freq, time)
        
        self.lstm_input_size = 128 * 5 # channels * freq_dim
        self.lstm = nn.LSTM(self.lstm_input_size, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        
        self.fc = nn.Linear(hidden_dim * 2, num_classes) # *2 for bidirectional
        
    def forward(self, x):
        # x shape: (batch, 1, n_mfcc, time_frames)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = self.adaptive_pool(x) # (batch, 128, 5, 20)
        
        # Prepare for LSTM: (batch, time, features)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, -1) # (batch, 20, 128*5)
        
        lstm_out, _ = self.lstm(x)
        # Use last time step
        x = lstm_out[:, -1, :] 
        x = self.fc(x)
        return x

# --- Model 2: Simple ResNet-SE Block ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResNetSEModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetSEModel, self).__init__()
        # Simplified ResNet-like structure with SE blocks
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), SEBlock(64)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), SEBlock(128)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

# --- Model 3: Simplified RawNet (1D Conv) ---
class RawNetModel(nn.Module):
    def __init__(self, num_classes=2):
        super(RawNetModel, self).__init__()
        # Takes raw audio (batch, 1, samples)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=128, stride=4) # Sinc-conv approximation
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(4)
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.LeakyReLU(0.2), nn.MaxPool1d(4)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.LeakyReLU(0.2), nn.MaxPool1d(4)
        )
        
        self.gru = nn.GRU(128, 256, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x shape: (batch, 1, samples)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        
        # (batch, channels, time) -> (batch, time, channels) for GRU
        x = x.permute(0, 2, 1) 
        _, h_n = self.gru(x)
        # h_n shape: (num_layers, batch, hidden_dim) -> take last layer
        x = h_n[-1] 
        x = self.fc(x)
        return x

# --- Ensemble Wrapper ---
class DeepfakeDetector:
    def __init__(self, weights_dir="models/weights"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model1 = CNNLSTMModel().to(self.device)
        self.model2 = ResNetSEModel().to(self.device)
        self.model3 = RawNetModel().to(self.device)
        
        self.weights_dir = weights_dir
        self._load_weights()
        
        self.model1.eval()
        self.model2.eval()
        self.model3.eval()

    def _load_weights(self):
        # In a real scenario, we would load state_dicts here.
        # For this prototype, we'll initialize with random weights if files don't exist.
        m1_path = os.path.join(self.weights_dir, "model1.pth")
        if os.path.exists(m1_path):
            self.model1.load_state_dict(torch.load(m1_path, map_location=self.device))
        else:
            logger.warning("Model 1 weights not found, using random init.")

        # Similarly for others...
        
    def predict(self, features, raw_audio):
        """
        features: dict containing 'mfcc', 'mel_spec', etc.
        raw_audio: numpy array of raw audio samples
        """
        with torch.no_grad():
            # Prepare inputs
            # 1. CNN-LSTM input (MFCCs as image)
            # MFCC shape from extractor: (n_mfcc, time)
            mfcc = torch.FloatTensor(features['mfcc']).unsqueeze(0).unsqueeze(0).to(self.device) # (1, 1, 40, T)
            p1 = F.softmax(self.model1(mfcc), dim=1)
            
            # 2. ResNet input (Mel-Spectrogram)
            mel = torch.FloatTensor(features['mel_spec']).unsqueeze(0).unsqueeze(0).to(self.device)
            # Resize mel to match expected input if needed, or rely on adaptive pool
            p2 = F.softmax(self.model2(mel), dim=1)
            
            # 3. RawNet input
            raw = torch.FloatTensor(raw_audio).unsqueeze(0).unsqueeze(0).to(self.device)
            p3 = F.softmax(self.model3(raw), dim=1)
            
            # Ensemble Voting (Soft Voting)
            # Weights can be tuned based on validation accuracy
            w1, w2, w3 = 0.4, 0.4, 0.2 
            avg_prob = (w1 * p1 + w2 * p2 + w3 * p3)
            
            # Class 1 = Fake, Class 0 = Real
            fake_prob = avg_prob[0][1].item()
            
            return {
                "is_deepfake": fake_prob > 0.5,
                "confidence": fake_prob,
                "individual_scores": {
                    "cnn_lstm": p1[0][1].item(),
                    "resnet_se": p2[0][1].item(),
                    "rawnet": p3[0][1].item()
                }
            }
