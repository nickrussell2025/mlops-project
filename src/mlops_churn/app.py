import logging
import os
import time
from datetime import datetime

os.environ["TZ"] = "Europe/London"
time.tzset()

import mlflow
import pandas as pd
from database import log_prediction
from flask import Flask, jsonify, request
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ValidationError

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(f"Python timezone: {time.tzname}, Current time: {datetime.now()}")


class CustomerData(BaseModel):
    """Basic validation for customer data"""

    CreditScore: float
    Gender: str
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float
    BalanceActivityInteraction: float
    ZeroBalance: float
    UnderUtilized: float
    AgeRisk: float
    GermanyRisk: float
    GermanyMatureCombo: float


def create_app():
    """Flask application factory"""

    app = Flask(__name__)

    # initialize MLflow
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    # load model on startup
    try:
        model_uri = f"models:/{config.MODEL_NAME}@{config.MODEL_ALIAS}"
        app.model = mlflow.sklearn.load_model(model_uri)
        app.model_loaded = True

        client = MlflowClient()
        mv = client.get_model_version_by_alias(config.MODEL_NAME, config.MODEL_ALIAS)
        app.model_version = mv.version

        print(f"✅ Model loaded: {model_uri} - version {app.model_version}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        app.model = None
        app.model_loaded = False
        app.model_version = "unknown"

    @app.route("/")
    def health_check():
        """Health check endpoint"""

        return jsonify(
            {
                "status": "healthy",
                "service": "bank-churn-prediction-api",
                "model_name": config.MODEL_NAME,
                "model_loaded": app.model_loaded,
                "timestamp": datetime.now().astimezone().isoformat(),
            }
        )

    @app.route("/predict", methods=["POST"])
    def predict():
        """Single prediction endpoint with Pydantic validation"""

        if not app.model_loaded:
            return jsonify({"error": "Model not loaded"}), 500

        try:
            request_data = request.get_json()
            if not request_data:
                return jsonify({"error": "No JSON data provided"}), 400

            # Validate with Pydantic
            try:
                customer_data = CustomerData(**request_data)
            except ValidationError as e:
                return jsonify({"error": "Invalid data", "details": str(e)}), 400

            # Convert and predict
            df_input = pd.DataFrame([customer_data.dict()])
            prediction = app.model.predict(df_input)[0]
            probability = app.model.predict_proba(df_input)[0][1]

            # Log to database
            model_version = getattr(app, "model_version", "unknown")
            log_success = log_prediction(
                customer_data.dict(), float(probability), model_version
            )

            if not log_success:
                logger.warning("Failed to log prediction to database")

            return jsonify(
                {
                    "prediction": int(prediction),
                    "probability": round(float(probability), 4),
                    "timestamp": datetime.now().astimezone().isoformat(),
                }
            )

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host=config.API_HOST, port=config.API_PORT, debug=True)
