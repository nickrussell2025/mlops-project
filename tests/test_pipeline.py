import pandas as pd

from src.mlops_churn.pipeline import ChurnDataProcessor, ChurnModelTrainer


def test_pipeline():
    """Test complete pipeline from data processing to prediction"""
    # Create test data
    test_data = pd.DataFrame(
        {
            "CreditScore": [600, 700, 650, 720, 580, 690, 610, 750, 640, 680, 620, 710],
            "Geography": ["France", "Spain", "Germany"] * 4,
            "Gender": ["Female", "Male"] * 6,
            "Age": [35, 45, 30, 50, 25, 40, 55, 28, 38, 42, 33, 47],
            "Tenure": [2, 5, 1, 8, 3, 6, 4, 7, 2, 5, 1, 8],
            "Balance": [
                0,
                50000,
                25000,
                75000,
                0,
                40000,
                60000,
                80000,
                0,
                30000,
                45000,
                55000,
            ],
            "NumOfProducts": [1, 2, 1, 3, 1, 2, 2, 3, 1, 2, 1, 3],
            "HasCrCard": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            "IsActiveMember": [1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
            "EstimatedSalary": [
                40000,
                60000,
                35000,
                75000,
                30000,
                55000,
                65000,
                80000,
                45000,
                50000,
                42000,
                58000,
            ],
            "Exited": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    # test data processing worked
    processor = ChurnDataProcessor()
    processor.df_clean = test_data
    processor.engineer_features()
    assert "high_value_customer" in processor.df_clean.columns
    assert "zero_balance" in processor.df_clean.columns

    # test features exist in data
    for feature in processor.categorical_features:
        assert feature in processor.df_clean.columns

    for feature in processor.numerical_features:
        assert feature in processor.df_clean.columns

    features, target = processor.get_features_and_target()

    # test model training
    trainer = ChurnModelTrainer()
    trainer.prepare_and_train(features, target, processor)

    assert trainer.pipeline is not None
    assert 0 <= trainer.accuracy <= 1
    assert len(trainer.y_train) > 0
    assert len(trainer.y_val) > 0
    assert len(trainer.y_test) > 0

    # test prediction
    prediction = trainer.pipeline.predict(features.head(1))

    # test prediction works
    assert prediction is not None
    assert len(prediction) == 1
