"""
Simple Flask application for testing containerized deployment.
Basic endpoints to validate container deployment works.
"""
import os
import time
from datetime import datetime

from flask import Flask, jsonify, request

# Set timezone
os.environ["TZ"] = "Europe/London"
time.tzset()

def create_app():
    """Flask application factory for container testing."""
    app = Flask(__name__)

    @app.route("/")
    def health_check():
        """Health check endpoint for container health checks."""
        return jsonify({
            "status": "healthy",
            "service": "container-test-api",
            "version": "1.0.0",
            "timestamp": datetime.now().astimezone().isoformat(),
            "container": "running",
            "server": "gunicorn"
        })

    @app.route("/predict", methods=["POST"])
    def mock_predict():
        """Mock prediction endpoint to test POST requests."""
        try:
            request_data = request.get_json()
            if not request_data:
                return jsonify({"error": "No JSON data provided"}), 400

            # Mock prediction response
            mock_result = {
                "prediction": 0.7234,
                "confidence": "high",
                "timestamp": datetime.now().astimezone().isoformat(),
                "model_version": "test-v1.0.0",
                "input_received": True,
                "container_deployment": "success"
            }

            return jsonify(mock_result)

        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    @app.route("/info")
    def info():
        """System info endpoint for debugging container deployment."""
        return jsonify({
            "python_version": "3.11",
            "flask_version": "3.1.1",
            "gunicorn": "running",
            "timezone": time.tzname[0],
            "container_test": "successful",
            "endpoints": ["/", "/predict", "/info"]
        })

    return app

# For development testing
if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)