import sys

import pandas as pd

sys.path.insert(0, 'src/mlops_churn')

from pipeline import ChurnModelTrainer


def test_prepare_training_data():

    fake_data = pd.DataFrame({
        'CreditScore': [619, 608, 502, 700, 750, 650, 720, 580, 690, 610],
        'Geography': ['France', 'Spain', 'France', 'Germany', 'Spain', 'France', 'Germany', 'Spain', 'France', 'Germany'],
        'Gender': ['Female', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
        'Age': [42, 41, 42, 35, 28, 45, 38, 52, 29, 33],
        'Tenure': [2, 1, 8, 5, 3, 6, 4, 7, 2, 5],
        'Balance': [0.0, 83807.86, 159660.8, 50000.0, 0.0, 75000.0, 0.0, 120000.0, 45000.0, 0.0],
        'NumOfProducts': [1, 1, 3, 2, 1, 2, 1, 3, 2, 1],
        'HasCrCard': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        'IsActiveMember': [1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
        'EstimatedSalary': [101348.88, 112542.58, 113931.57, 75000.0, 45000.0, 95000.0, 68000.0, 125000.0, 55000.0, 78000.0],
        'zero_balance': [1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
        'high_value_customer': [0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
        'Exited': [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]
    })

    trainer = ChurnModelTrainer()
    categorical_features = ["Geography", "Gender"]
    numerical_features = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "zero_balance",
        "high_value_customer",
    ]

    trainer.prepare_training_data(fake_data, categorical_features, numerical_features)

    assert trainer.X_train_df is not None
    assert trainer.X_val_df is not None
    assert trainer.X_test_df is not None
    assert len(trainer.y_train) > 0
    assert len(trainer.y_val) > 0
    assert len(trainer.y_test) > 0
