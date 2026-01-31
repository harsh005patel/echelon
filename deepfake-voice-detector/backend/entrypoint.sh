#!/bin/bash

# Check if model weights exist
if [ ! -f "models/weights/model1.pth" ]; then
    echo "Model weights not found. Generating mock data and training..."
    python -m training.generate_mock_data
    python -m training.train_model
else
    echo "Model weights found. Skipping training."
fi

# Start the server
echo "Starting FastAPI server..."
exec uvicorn app:app --host 0.0.0.0 --port 8000
