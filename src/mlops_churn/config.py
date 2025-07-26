import os
from pathlib import Path


class Config:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"

    MODEL_NAME = os.getenv("MODEL_NAME", "bank-churn-classifier")

    # Flask API config
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8080"))

    # Database config
    DATABASE_HOST = os.getenv("DATABASE_HOST", "localhost")
    DATABASE_PORT = os.getenv("DATABASE_PORT", "5432")
    DATABASE_NAME = os.getenv("DATABASE_NAME", "monitoring")
    DATABASE_USER = os.getenv("DATABASE_USER", "postgres")
    DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD", "example")

    # Storage config
    USE_CLOUD_STORAGE = os.getenv("USE_CLOUD_STORAGE", "false").lower() == "true"
    BUCKET_NAME = os.getenv("BUCKET_NAME", "local-mlflow-artifacts")

    # Reference data paths
    LOCAL_REFERENCE_PATH = os.getenv(
        "LOCAL_REFERENCE_PATH", "monitoring/reference_data.parquet"
    )
    CLOUD_REFERENCE_PATH = os.getenv(
        "CLOUD_REFERENCE_PATH", "reference/reference_data.parquet"
    )


config = Config()
