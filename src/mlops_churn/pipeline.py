import os

import mlflow
import mlflow.sklearn
import pandas as pd

# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
from prefect import flow, task
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import config


class ChurnDataProcessor:
    def __init__(self, data_path=None):
        self.data_path = data_path or str(config.DATA_DIR / "raw")
        self.df_clean = None
        self.balance_threshold = None

    def load_and_clean_data(self):
        """Load CSV, fix data types, remove unnecessary columns"""

        # check if file exists
        file_path = f"{self.data_path}/Churn_Modelling.csv"

        if not os.path.exists(file_path):
            print("file not found...downloading")
            try:
                os.system(
                    f"kaggle datasets download -d shrutimechlearn/churn-modelling -p {self.data_path} --unzip"
                )
                print("data downloaded successfully")
            except Exception as e:
                print(f"download failed {e}")
                raise
        else:
            print("file exists, skipping download")

        df = pd.read_csv(file_path)
        print(f"loaded {len(df)} records")

        df_clean = df.copy()

        df_clean = df_clean[
            [
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
        ]

        df_clean["Geography"] = df_clean["Geography"].astype("category")
        df_clean["Gender"] = df_clean["Gender"].astype("category")

        numerical_features = ["CreditScore", "Age", "Tenure", "NumOfProducts"]

        for col in numerical_features:
            df_clean[col] = df_clean[col].astype("float64")

        self.df_clean = df_clean

        return self

    def engineer_features(self):
        """Create features from dataset and add new engineered financial and risk features"""

        # calculate threshold
        self.balance_threshold = self.df_clean["Balance"].quantile(0.75)

        # create new augmented features
        self.df_clean["high_value_customer"] = (
            self.df_clean["Balance"] > self.balance_threshold
        ).astype(int)
        self.df_clean["zero_balance"] = (self.df_clean["Balance"] == 0).astype(int)

        # define feature sets
        self.categorical_features = ["Geography", "Gender"]
        self.numerical_features = [
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

        # save reference data for monitoring
        self.save_reference_data()

        print(
            f"Feature engineering completed: {len(self.categorical_features)} categorical features, {len(self.numerical_features)} numerical features"
        )

        return self

    def save_reference_data(self):
        """Save reference dataset for monitoring"""

        os.makedirs("monitoring", exist_ok=True)

        # save reference data (training data without target)
        reference_data = self.df_clean.drop("Exited", axis=1).copy()
        reference_data.to_parquet("monitoring/reference_data.parquet")

        print(f"Reference data saved: {len(reference_data)} records")

        return reference_data


class ChurnModelTrainer:
    def __init__(self):
        self.X_train_df = None
        self.X_val_df = None
        self.X_test_df = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.dv = None
        self.best_model = None
        self.best_model_name = None
        self.best_threshold = None
        self.best_f1 = None

    def prepare_training_data(self, df_clean, categorical_features, numerical_features):
        """Split dataset into training/validation/test"""

        # split data
        df_full_train, df_test = train_test_split(
            df_clean, test_size=0.2, random_state=1, stratify=df_clean["Exited"]
        )
        df_train, df_val = train_test_split(
            df_full_train,
            test_size=0.25,
            random_state=1,
            stratify=df_full_train["Exited"],
        )

        # reset indices
        df_train = df_train.reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)

        # create target variables
        y_train = df_train["Exited"].values
        y_val = df_val["Exited"].values
        y_test = df_test["Exited"].values

        # remove target from features
        df_train = df_train.drop("Exited", axis=1)
        df_val = df_val.drop("Exited", axis=1)
        df_test = df_test.drop("Exited", axis=1)

        # apply transformation
        dv = DictVectorizer(sparse=False)
        train_dict = df_train[categorical_features + numerical_features].to_dict(
            orient="records"
        )
        X_train = dv.fit_transform(train_dict)

        val_dict = df_val[categorical_features + numerical_features].to_dict(
            orient="records"
        )
        X_val = dv.transform(val_dict)

        test_dict = df_test[categorical_features + numerical_features].to_dict(
            orient="records"
        )
        X_test = dv.transform(test_dict)

        # convert to DataFrames
        feature_names = dv.get_feature_names_out()
        self.X_train_df = pd.DataFrame(X_train, columns=feature_names)
        self.X_val_df = pd.DataFrame(X_val, columns=feature_names)
        self.X_test_df = pd.DataFrame(X_test, columns=feature_names)
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.dv = dv

        print(
            f"✅ Training data prepared: {len(self.X_train_df)} train, {len(self.X_val_df)} val, {len(self.X_test_df)} test"
        )

        return self

    def train_model(self, processor):
        """Train basic logistic regression model"""

        with mlflow.start_run():
            # train model
            model = LogisticRegression(random_state=42, max_iter=10000)
            model.fit(self.X_train_df, self.y_train)

            # evaluate on validation set
            y_pred = model.predict(self.X_val_df)
            accuracy = accuracy_score(self.y_val, y_pred)
            f1 = f1_score(self.y_val, y_pred)

            # log to MLflow
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("balance_threshold", processor.balance_threshold)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)

            # save model with signature and input example
            from mlflow.models.signature import infer_signature

            signature = infer_signature(self.X_train_df, model.predict(self.X_train_df))
            input_example = self.X_train_df.head(1)

            logged_model = mlflow.sklearn.log_model(
                model, name="model", signature=signature, input_example=input_example
            )
            self.model_uri = logged_model.model_uri

            print(f"✅ Model trained: Accuracy={accuracy:.3f}, F1={f1:.3f}")

            self.best_model = model
            self.best_f1 = f1

            return self


class ChurnModelRegistry:
    def register_model(self, trainer, processor):
        """Register complete pipeline in MLflow registry"""

        # create complete pipeline
        pipeline = Pipeline(
            [("vectorizer", trainer.dv), ("classifier", trainer.best_model)]
        )

        with mlflow.start_run():
            # log parameters
            mlflow.log_params(
                {
                    "model_type": "LogisticRegression",
                    "balance_threshold": processor.balance_threshold,
                    "train_size": len(trainer.X_train_df),
                }
            )

            # log complete pipeline
            from mlflow.models.signature import infer_signature

            signature = infer_signature(
                trainer.X_train_df, trainer.best_model.predict(trainer.X_train_df)
            )

            logged_model = mlflow.sklearn.log_model(
                pipeline, name="model", signature=signature
            )

            # register and set alias
            registered_model = mlflow.register_model(
                logged_model.model_uri, config.MODEL_NAME
            )

            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            client.set_registered_model_alias(
                name=config.MODEL_NAME,
                alias="production",
                version=registered_model.version,
            )

            print(
                f"✅ Complete pipeline registered: version {registered_model.version}"
            )
            return registered_model


@task
def data_processing():
    """Data processing task"""

    processor = ChurnDataProcessor()
    processor.load_and_clean_data()
    processor.engineer_features()

    return processor


@task
def model_training(processor):
    """Model training task"""

    trainer = ChurnModelTrainer()
    trainer.prepare_training_data(
        processor.df_clean, processor.categorical_features, processor.numerical_features
    )
    trainer.train_model(processor)

    return trainer


@task
def register_model(trainer, processor):
    """Model registry task"""

    registry = ChurnModelRegistry()
    registered_model = registry.register_model(trainer, processor)

    return registered_model


@flow
def pipeline():
    """Main pipeline flow"""

    print("testing prefect flow")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("bank-churn-prediction")

    print("MLflow configured")

    processor = data_processing()
    trainer = model_training(processor)
    registered_model = register_model(trainer, processor)

    print("flow completed")

    return registered_model


if __name__ == "__main__":
    pipeline()
