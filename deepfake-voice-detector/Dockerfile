# Backend Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (for audio libraries)
RUN apt-get update && apt-get install -y \
    gcc \
    libsndfile1 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .
# Create directories for models/logs if not exist
RUN mkdir -p models/weights logs

EXPOSE 8000

# Copy entrypoint and make executable
COPY backend/entrypoint.sh .
RUN chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]
