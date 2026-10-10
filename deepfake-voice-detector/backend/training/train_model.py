import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.deepfake_detector import CNNLSTMModel, ResNetSEModel, RawNetModel
from training.dataset_loader import DeepfakeDataset
import os
from loguru import logger

def train(epochs=12, batch_size=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training ensemble models using device: {device}")
    
    # Check data
    data_dir = "data/raw"
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        logger.error("No data found! Run generate_mock_data.py first.")
        return

    # Dataset
    dataset = DeepfakeDataset(data_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    # Models
    model1 = CNNLSTMModel().to(device)
    model2 = ResNetSEModel().to(device)
    model3 = RawNetModel().to(device)
    
    # Optimizers with weight decay for regularization and numerical stability
    opt1 = optim.AdamW(model1.parameters(), lr=0.0008, weight_decay=1e-4)
    opt2 = optim.AdamW(model2.parameters(), lr=0.0005, weight_decay=1e-4)
    opt3 = optim.AdamW(model3.parameters(), lr=0.0003, weight_decay=1e-4)
    
    criterion = nn.CrossEntropyLoss()
    
    # Save directories
    primary_dir = "models/weights"
    alt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
    os.makedirs(primary_dir, exist_ok=True)
    os.makedirs(alt_dir, exist_ok=True)
    
    model1.train()
    model2.train()
    model3.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct1, correct2, correct3, total_samples = 0, 0, 0, 0

        for batch in dataloader:
            labels = batch['label'].squeeze(1).to(device)
            bsz = labels.size(0)
            total_samples += bsz

            # 1. Train Model 1 (CNN-LSTM on MFCC)
            # shape: (batch, 1, 40, T)
            mfcc = batch['mfcc'].unsqueeze(1).to(device)
            opt1.zero_grad()
            out1 = model1(mfcc)
            loss1 = criterion(out1, labels)
            loss1.backward()
            torch.nn.utils.clip_grad_norm_(model1.parameters(), 2.0)
            opt1.step()
            correct1 += (out1.argmax(1) == labels).sum().item()

            # 2. Train Model 2 (ResNet on Mel)
            # shape: (batch, 1, 128, T)
            mel = batch['mel_spec'].unsqueeze(1).to(device)
            opt2.zero_grad()
            out2 = model2(mel)
            loss2 = criterion(out2, labels)
            loss2.backward()
            torch.nn.utils.clip_grad_norm_(model2.parameters(), 2.0)
            opt2.step()
            correct2 += (out2.argmax(1) == labels).sum().item()

            # 3. Train Model 3 (RawNet on Audio)
            # shape: (batch, 1, samples)
            audio = batch['audio'].unsqueeze(1).to(device)
            opt3.zero_grad()
            out3 = model3(audio)
            loss3 = criterion(out3, labels)
            loss3.backward()
            torch.nn.utils.clip_grad_norm_(model3.parameters(), 2.0)
            opt3.step()
            correct3 += (out3.argmax(1) == labels).sum().item()

            total_loss += (loss1.item() + loss2.item() + loss3.item())

        acc1 = (correct1 / total_samples) * 100
        acc2 = (correct2 / total_samples) * 100
        acc3 = (correct3 / total_samples) * 100
        logger.info(f"Epoch [{epoch+1:02d}/{epochs:02d}] Loss: {total_loss:.4f} | Acc M1: {acc1:.1f}% | M2(ResNet): {acc2:.1f}% | M3(RawNet): {acc3:.1f}%")

    # Save to both locations
    for save_path in [primary_dir, alt_dir]:
        torch.save(model1.state_dict(), os.path.join(save_path, "model1.pth"))
        torch.save(model2.state_dict(), os.path.join(save_path, "model2.pth"))
        torch.save(model3.state_dict(), os.path.join(save_path, "model3.pth"))
        logger.info(f"Saved checkpoints to: {save_path}")

    logger.info("Training complete. All 3 models saved successfully.")

if __name__ == "__main__":
    train()
