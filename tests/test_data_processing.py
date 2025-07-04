import sys

import pandas as pd

sys.path.insert(0, "src/mlops_churn")

from src.mlops_churn.pipeline import ChurnDataProcessor


def test_load_and_clean_data():
    """Test data loading and cleaning"""

    processor = ChurnDataProcessor()
    processor.load_and_clean_data()

    # check data was loaded
    assert len(processor.df_clean) > 0

    # check required columns exist
    required_cols = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    for col in required_cols:
        assert col in processor.df_clean.columns

    # test datatypes
    assert processor.df_clean["CreditScore"].dtype == "float64"
    assert processor.df_clean["Geography"].dtype.name == "category"
    assert processor.df_clean["Gender"].dtype.name == "category"
    assert processor.df_clean["Age"].dtype == "float64"
    assert processor.df_clean["Tenure"].dtype == "float64"
    assert processor.df_clean["Balance"].dtype == "float64"
    assert processor.df_clean["NumOfProducts"].dtype == "float64"
    assert processor.df_clean["HasCrCard"].dtype == "float64"
    assert processor.df_clean["IsActiveMember"].dtype == "float64"
    assert processor.df_clean["EstimatedSalary"].dtype == "float64"
    assert processor.df_clean["Exited"].dtype == "int64"


def test_engineer_features():
    """Test feature engineering creates required features"""

    # create sample data with required columns
    sample_data = pd.DataFrame(
        {
            "CreditScore": [600, 700, 800],
            "Geography": ["France", "Germany", "Spain"],
            "Gender": ["Male", "Female", "Male"],
            "Age": [25, 35, 45],
            "Tenure": [1, 5, 10],
            "Balance": [0.0, 50000.0, 100000.0],
            "NumOfProducts": [1, 2, 3],
            "HasCrCard": [0, 1, 1],
            "IsActiveMember": [1, 0, 1],
            "EstimatedSalary": [30000, 50000, 70000],
            "Exited": [0, 1, 0],
        }
    )

    processor = ChurnDataProcessor()
    processor.df_clean = sample_data
    processor.engineer_features()

    # test new features were created
    assert "high_value_customer" in processor.df_clean.columns
    assert "zero_balance" in processor.df_clean.columns

    # test feature lists were set
    assert processor.categorical_features == ["Geography", "Gender"]
    assert len(processor.numerical_features) == 10
