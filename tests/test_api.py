import requests

BASE_URL = "http://localhost:8080"


def test_health_endpoint():
    """Test health endpoint call"""

    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_predict_endpoint():
    """Test prediction endpoint call with dummy data"""

    test_data = {
        "Geography": "France",
        "Gender": "Female",
        "CreditScore": 619,
        "Age": 42,
        "Tenure": 2,
        "Balance": 0.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000.0,
        "zero_balance": 1,
        "high_value_customer": 0,
    }

    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "probability" in data
    assert "prediction" in data
