"""Configuration for the MLOps project."""

import os
from pathlib import Path


class Config:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"

    MODEL_NAME = "bank-churn-classifier"

    # flask api config
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8080"))


config = Config()
