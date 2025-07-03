import sys
sys.path.append('src')
from mlops_churn.database import log_prediction, log_drift_result

# Test prediction logging
test_data = {"CreditScore": 600, "Age": 40, "Geography": "France"}
result = log_prediction(test_data, 0.25, "v6")
print(f"Prediction logged: {result}")

# Test drift logging
drift_result = log_drift_result(True, 0.8, "Age", "data_drift")
print(f"Drift logged: {drift_result}")
