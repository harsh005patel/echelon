import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.deepfake_detector import CNNLSTMModel, ResNetSEModel, RawNetModel
from training.dataset_loader import DeepfakeDataset
import os
from loguru import logger

def train(epochs=5, batch_size=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Check data
    if not os.path.exists("data/raw") or not os.listdir("data/raw"):
        logger.error("No data found! Run generate_mock_data.py first.")
        return

    # Dataset
    dataset = DeepfakeDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Models
    model1 = CNNLSTMModel().to(device)
    model2 = ResNetSEModel().to(device)
    model3 = RawNetModel().to(device)
    
    # Optimizers
    opt1 = optim.Adam(model1.parameters(), lr=0.001)
    opt2 = optim.Adam(model2.parameters(), lr=0.001)
    opt3 = optim.Adam(model3.parameters(), lr=0.0001)
    
    criterion = nn.CrossEntropyLoss()
    
    save_dir = "models/weights"
    os.makedirs(save_dir, exist_ok=True)
    
    model1.train()
    model2.train()
    model3.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            labels = batch['label'].squeeze().to(device)
            # Handle batch size 1 edge case for BatchNorm
            if labels.dim() == 0: 
                 labels = labels.unsqueeze(0)
            
            # 1. Train Model 1 (CNN-LSTM on MFCC)
            mfcc = batch['mfcc'].unsqueeze(1).to(device) # Add channel dim
            opt1.zero_grad()
            out1 = model1(mfcc)
            loss1 = criterion(out1, labels)
            loss1.backward()
            opt1.step()
            
            # 2. Train Model 2 (ResNet on Mel)
            mel = batch['mel_spec'].unsqueeze(1).to(device)
            opt2.zero_grad()
            out2 = model2(mel)
            loss2 = criterion(out2, labels)
            loss2.backward()
            opt2.step()
            
            # 3. Train Model 3 (RawNet on Audio)
            audio = batch['audio'].unsqueeze(1).to(device)
            opt3.zero_grad()
            out3 = model3(audio)
            loss3 = criterion(out3, labels)
            loss3.backward()
            opt3.step()
            
            total_loss += (loss1.item() + loss2.item() + loss3.item())
            
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")
        
    # Save Weights
    torch.save(model1.state_dict(), os.path.join(save_dir, "model1.pth"))
    torch.save(model2.state_dict(), os.path.join(save_dir, "model2.pth"))
    torch.save(model3.state_dict(), os.path.join(save_dir, "model3.pth"))
    logger.info("Training complete. Models saved.")

if __name__ == "__main__":
    train()
