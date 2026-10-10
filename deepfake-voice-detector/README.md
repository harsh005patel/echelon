# Deepfake Voice Detector

Real-time audio phishing and deepfake voice detection using an ensemble of deep learning models.

---

## Architecture Overview

```
                      ┌──────────────────────┐
                      │   Incoming Audio     │
                      │ (Live Mic / Upload)  │
                      └──────────┬───────────┘
                                 │
                     ┌───────────▼───────────┐
                     │   Audio Preprocessor  │
                     │  (Noise Reduce, Norm) │
                     └───────────┬───────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           ▼                     ▼                     ▼
┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐
│      Model 1       │ │      Model 2       │ │      Model 3       │
│      CNN-LSTM      │ │     ResNet-SE      │ │       RawNet       │
│  (Input: MFCC 40)  │ │ (Input: Mel Spec)  │ │ (Input: Raw Audio) │
│    Weight: 40%     │ │    Weight: 40%     │ │    Weight: 20%     │
└──────────┬─────────┘ └─────────┬──────────┘ └─────────┬──────────┘
           │                     │                      │
           └─────────────────────┼──────────────────────┘
                                 │
                     ┌───────────▼───────────┐
                     │   Ensemble Voting     │
                     │  (Soft-Probability)   │
                     └───────────┬───────────┘
                                 │
                     ┌───────────▼───────────┐
                     │  Prediction & Alert   │
                     │  (Real vs. Deepfake)  │
                     └───────────────────────┘
```

---

## Model Weights & Auto-Discovery

The detector automatically searches for `model1.pth`, `model2.pth`, and `model3.pth` in the following locations (both local dev and Docker container):
- Explicitly passed `--weights_dir`
- `models/weights/`
- `models/` (project root)
- `../models/`
- `/app/models/weights` (Docker mount)

### State-of-the-Art Pretrained Checkpoints Across the Web
For production accuracy against modern voice cloning (ElevenLabs, Tortoise, VALL-E):
1. **AASIST / AASIST-L (ASVspoof 2019/2021 LA Challenge Winner)**
   - Repository: [Clova AI AASIST](https://github.com/clovaai/aasist)
   - Pretrained weights: Available in repository releases (`AASIST.pth`, `AASIST-L.pth`)
2. **Spectra-AASIST & AASIST3 (Hugging Face)**
   - Hugging Face: [lab260/spectra_aasist](https://huggingface.co/lab260/spectra_aasist) and [lab260/AASIST3](https://huggingface.co/lab260/AASIST3)
3. **RawNet2 Anti-Spoofing**
   - Repository: [asvspoof-2021/rawnet2](https://github.com/eurecom-asp/RawNet2)
4. **SpeechBrain Spoofing Detection**
   - SpeechBrain recipe: `speechbrain/spkrec-ecapa-voxceleb` / ASVspoof recipes

---

## Running the Application

### 1. Backend (FastAPI + WebSockets + PyTorch)
```bash
cd deepfake-voice-detector/backend
pip install -r requirements.txt
python app.py
```
- Health Check: `http://localhost:8000/health`
- WebSocket Endpoint: `ws://localhost:8000/ws/audio`
- File Analysis REST API: `POST http://localhost:8000/api/analyze`

### 2. CLI Prediction on Audio Files
```bash
cd deepfake-voice-detector/backend
python predict_cli.py path/to/sample.wav
```

### 3. Frontend (React + Vite)
```bash
cd deepfake-voice-detector/frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## Features
- **Real-Time Audio Spectrum Visualizer**: Dynamic HTML5 Canvas audio frequency display.
- **Microphone Stream Analysis**: Low-latency 16kHz PCM streaming via WebSocket.
- **Direct Audio File Upload**: Upload `.wav`, `.mp3`, `.ogg`, or `.flac` files for immediate classification.
- **Triple-Model Breakdown**: Detailed confidence score and individual model probabilities (CNN-LSTM, ResNet-SE, RawNet).
- **History & Alert Log**: Chronological timeline of detections with latency in milliseconds.
