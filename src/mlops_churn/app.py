import os
import sys

import mlflow
from flask import Flask, jsonify, request

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from datetime import datetime

from config import config


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
        print(f"✅ Model loaded: {model_uri}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        app.model = None
        app.model_loaded = False

    @app.route("/")
    def health_check():
        """Health check endpoint"""
        return jsonify(
            {
                "status": "healthy",
                "service": "bank-churn-prediction-api",
                "model_name": config.MODEL_NAME,
                "model_loaded": app.model_loaded,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    @app.route("/predict", methods=["POST"])
    def predict():
        """Single prediction endpoint"""
        if not app.model_loaded:
            return jsonify({"error": "Model not loaded"}), 500

        try:
            # Get input data
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            # Make prediction
            prediction = app.model.predict([data])[0]
            probability = app.model.predict_proba([data])[0][1]

            return jsonify(
                {
                    "prediction": int(prediction),
                    "probability": float(probability),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host=config.API_HOST, port=config.API_PORT, debug=True)
