"""Configuration settings for TerraTech application."""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.resolve()
APP_DIR = BASE_DIR / "app"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = MODELS_DIR

# Flask settings
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"

# Model paths
UPLOAD_FOLDER = BASE_DIR / "uploads"
MODEL_FOLDER = MODELS_DIR
DATA_PATH = MODELS_DIR / "data.csv"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.h5"
XGB_MODEL_PATH = MODELS_DIR / "xgb_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
MODEL_TYPE_PATH = MODELS_DIR / "model_type.txt"

# Static and templates paths (absolute paths)
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Create required directories
UPLOAD_FOLDER.mkdir(exist_ok=True)