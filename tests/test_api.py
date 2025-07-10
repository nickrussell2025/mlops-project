import pytest
import requests

BASE_URL = "http://localhost:8080"


@pytest.fixture
def valid_customer_data():
    """Fixture for valid test data"""
    return {
        "CreditScore": 619.0,
        "Gender": "Female",
        "Age": 42.0,
        "Tenure": 2.0,
        "Balance": 0.0,
        "NumOfProducts": 1.0,
        "HasCrCard": 1.0,
        "IsActiveMember": 1.0,
        "EstimatedSalary": 50000.0,
        "BalanceActivityInteraction": 0.0,
        "ZeroBalance": 1.0,
        "UnderUtilized": 1.0,
        "AgeRisk": 0.0,
        "GermanyRisk": 0.0,
        "GermanyMatureCombo": 0.0,
    }


def test_health_endpoint():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_endpoint(valid_customer_data):
    """Test prediction endpoint"""
    response = requests.post(f"{BASE_URL}/predict", json=valid_customer_data)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_missing_fields():
    """Test prediction with missing fields"""
    invalid_data = {
        "CreditScore": 619.0,
        "Gender": "Female",
        # Missing other required fields
    }

    response = requests.post(f"{BASE_URL}/predict", json=invalid_data)
    assert response.status_code == 400
    assert "error" in response.json()


def test_predict_no_data():
    """Test prediction with empty JSON data"""
    response = requests.post(f"{BASE_URL}/predict", json={})
    assert response.status_code == 400
    assert "error" in response.json()
