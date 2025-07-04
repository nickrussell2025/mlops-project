import os

import mlflow
import mlflow.sklearn
import pandas as pd

# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
from prefect import flow, task
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ChurnDataProcessor:
    def __init__(self, data_path="data/raw"):
        self.data_path = data_path
        self.df_clean = None

    def load_and_clean_data(self):
        """Load CSV and clean data types"""

        file_path = f"{self.data_path}/Churn_Modelling.csv"

        if not os.path.exists(file_path):
            print("Downloading data...")
            os.makedirs(self.data_path, exist_ok=True)
            os.system(
                f"kaggle datasets download -d shrutimechlearn/churn-modelling -p {self.data_path} --unzip"
            )

        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records")

        # select only needed columns
        columns_to_keep = [
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
        self.df_clean = df[columns_to_keep].copy()

        # fix data types
        self.df_clean["Geography"] = self.df_clean["Geography"].astype("category")
        self.df_clean["Gender"] = self.df_clean["Gender"].astype("category")

        numerical_cols = [
            "CreditScore",
            "Age",
            "Tenure",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
            "Balance",
        ]
        for col in numerical_cols:
            self.df_clean[col] = self.df_clean[col].astype("float64")

        return self

    def engineer_features(self):
        """Add engineered features"""
        
        # define feature lists
        self.categorical_features = ["Geography", "Gender"]
        
        balance_threshold = self.df_clean["Balance"].quantile(0.75)
        self.df_clean["high_value_customer"] = (self.df_clean["Balance"] > balance_threshold).astype(float)
        self.df_clean["zero_balance"] = (self.df_clean["Balance"] == 0).astype(float)
        
        # calculate numerical features after engineering
        self.numerical_features = [col for col in self.df_clean.columns 
                                if col not in self.categorical_features + ["Exited"]]
        
        print("Feature engineering completed")
        return self

    def save_reference_data(self):
        """Save reference data for monitoring"""

        os.makedirs("monitoring", exist_ok=True)
        reference_data = self.df_clean.drop("Exited", axis=1)
        reference_data.to_parquet("monitoring/reference_data.parquet")
        print(f"Reference data saved: {len(reference_data)} records")
        return reference_data

    def get_features_and_target(self):
        """Get features and target for training"""

        features = self.df_clean.drop("Exited", axis=1)
        target = self.df_clean["Exited"]
        return features, target


class ChurnModelTrainer:
    def __init__(self):
        self.pipeline = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def prepare_and_train(self, features, target, processor):
        """Split data and train complete pipeline"""

        # split data
        X_full_train, self.X_test, y_full_train, self.y_test = train_test_split(
            features, target, test_size=0.2, random_state=1, stratify=target
        )

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_full_train,
            y_full_train,
            test_size=0.25,
            random_state=1,
            stratify=y_full_train,
        )

        categorical_features = processor.categorical_features
        numerical_features = processor.numerical_features

        # create preprocessing pipeline
        preprocessor = ColumnTransformer(
            [
                (
                    "cat",
                    OneHotEncoder(drop="first", handle_unknown="ignore"),
                    categorical_features,
                ),
                ("num", StandardScaler(), numerical_features),
            ]
        )

        # create complete pipeline
        self.pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        random_state=1, max_iter=20000, class_weight="balanced"
                    ),
                ),
            ]
        )

        # train pipeline
        self.pipeline.fit(self.X_train, self.y_train)

        # evaluate
        y_pred = self.pipeline.predict(self.X_val)
        accuracy = accuracy_score(self.y_val, y_pred)
        f1 = f1_score(self.y_val, y_pred)

        print(f"Model trained: Accuracy={accuracy:.3f}, F1={f1:.3f}")
        print(
            f"Data split: {len(self.X_train)} train, {len(self.X_val)} val, {len(self.X_test)} test"
        )

        return self

    def log_to_mlflow(self):
        """Log pipeline to MLflow"""

        with mlflow.start_run():
            # calculate metrics
            y_pred = self.pipeline.predict(self.X_val)
            accuracy = accuracy_score(self.y_val, y_pred)
            f1 = f1_score(self.y_val, y_pred)

            # log parameters and metrics
            mlflow.log_params(
                {
                    "model_type": "Pipeline_LogisticRegression",
                    "train_size": len(self.X_train),
                    "n_features": self.X_train.shape[1],
                }
            )

            mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1})

            # log pipeline with signature
            from mlflow.models.signature import infer_signature

            signature = infer_signature(
                self.X_train, self.pipeline.predict(self.X_train)
            )

            logged_model = mlflow.sklearn.log_model(
                self.pipeline,
                name="model",
                signature=signature,
                input_example=self.X_train.head(1),
            )

            print("Model logged to MLflow")
            return logged_model


class ChurnModelRegistry:
    def __init__(self, model_name="bank-churn-classifier"):
        self.model_name = model_name

    def register_model(self, trainer):
        """Register trained pipeline in MLflow registry"""

        with mlflow.start_run():
            # test dataset evaluation
            y_test_pred = trainer.pipeline.predict(trainer.X_test)
            test_accuracy = accuracy_score(trainer.y_test, y_test_pred)
            test_f1 = f1_score(trainer.y_test, y_test_pred)

            # log final test metrics
            mlflow.log_params(
                {
                    "model_type": "Pipeline_LogisticRegression",
                    "test_size": len(trainer.X_test),
                    "final_model": True,
                }
            )

            mlflow.log_metrics({"test_accuracy": test_accuracy, "test_f1": test_f1})

            # log the pipeline
            from mlflow.models.signature import infer_signature

            signature = infer_signature(
                trainer.X_train, trainer.pipeline.predict(trainer.X_train)
            )

            logged_model = mlflow.sklearn.log_model(
                trainer.pipeline, name="model", signature=signature
            )

            # register model
            registered_model = mlflow.register_model(
                logged_model.model_uri, self.model_name
            )

            # set production alias
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            client.set_registered_model_alias(
                name=self.model_name,
                alias="production",
                version=registered_model.version,
            )

            print(
                f"Pipeline registered: {self.model_name} version {registered_model.version}"
            )
            print(f"Test performance: Accuracy={test_accuracy:.3f}, F1={test_f1:.3f}")

            return registered_model


@task
def data_processing():
    """Data processing task"""

    processor = ChurnDataProcessor()
    processor.load_and_clean_data()
    processor.engineer_features()
    processor.save_reference_data()
    features, target = processor.get_features_and_target()

    return features, target, processor


@task
def model_training(features, target, processor):
    """Model training task"""

    trainer = ChurnModelTrainer()
    trainer.prepare_and_train(features, target, processor)
    trainer.log_to_mlflow()

    return trainer


@task
def register_model(trainer):
    """Model registry task"""

    registry = ChurnModelRegistry()
    registered_model = registry.register_model(trainer)

    return registered_model


@flow
def pipeline():
    """Main pipeline flow"""

    print("testing prefect flow")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("bank-churn-prediction")

    print("MLflow configured")

    features, target, processor = data_processing()
    trainer = model_training(features, target, processor)
    registered_model = register_model(trainer)

    print("Pipeline completed")

    return registered_model


if __name__ == "__main__":
    pipeline()
