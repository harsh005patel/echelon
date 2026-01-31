import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Deepfake Voice Detector"
    API_V1_STR: str = "/api"
    SECRET_KEY: str = "CHANGE_THIS_IN_PRODUCTION"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Audio Settings
    SAMPLE_RATE: int = 16000
    CHUNK_DURATION: int = 3  # seconds
    OVERLAP: float = 0.5     # seconds
    
    # Model Settings
    MODEL_PATH: str = os.path.join("models", "weights")
    
    class Config:
        env_file = ".env"

settings = Settings()
