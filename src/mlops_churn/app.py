import os
import sys
import time
from datetime import datetime

os.environ["TZ"] = "Europe/London"
time.tzset()

import mlflow
import pandas as pd
from flask import Flask, jsonify, request
from mlflow.tracking import MlflowClient

from src.mlops_churn.config import config
from src.mlops_churn.database import log_prediction

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

print(f"Python timezone: {time.tzname}, Current time: {datetime.now()}")


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
        print(f"❌ Failed to load model: {e}")
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
        """Single prediction endpoint"""

        if not app.model_loaded:
            return jsonify({"error": "Model not loaded"}), 500

        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            df_input = pd.DataFrame([data])
            prediction = app.model.predict(df_input)[0]
            probability = app.model.predict_proba(df_input)[0][1]

            model_version = getattr(app, "model_version", "unknown")
            log_success = log_prediction(data, float(probability), model_version)

            if not log_success:
                print("Warning: failed to log prediction to database")

            return jsonify(
                {
                    "prediction": int(prediction),
                    "probability": round(float(probability), 4),
                    "timestamp": datetime.now().astimezone().isoformat(),
                }
            )

        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host=config.API_HOST, port=config.API_PORT, debug=True)
