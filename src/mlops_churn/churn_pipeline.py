import os
import time

os.environ["TZ"] = "Europe/London"
time.tzset()

import mlflow
import mlflow.sklearn
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from prefect import flow, task
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


@task
def setup_mlflow():
    """Initialize MLflow with minimal logging to reduce clutter."""
    mlflow_port = os.getenv("MLFLOW_PORT", "5000")
    mlflow.set_tracking_uri(f"http://localhost:{mlflow_port}")
    mlflow.set_experiment("bank-churn-prediction")

    mlflow.sklearn.autolog(
        log_input_examples=False,
        log_model_signatures=True,
        log_models=False,
        max_tuning_runs=0,
        silent=True,
        log_post_training_metrics=False,
    )

    print(f"✅ MLflow configured: http://localhost:{mlflow_port}")
    return mlflow.get_experiment_by_name("bank-churn-prediction").experiment_id


@task
def load_and_prepare_data(df):
    """Load data and create optimized features."""
    # Essential columns
    cols = [
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
    clean_df = df[cols].copy()

    # Data types
    clean_df["Geography"] = clean_df["Geography"].astype("category")
    clean_df["Gender"] = clean_df["Gender"].astype("category")
    numeric_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    for col in numeric_cols:
        clean_df[col] = clean_df[col].astype("float64")

    # Feature engineering
    clean_df["BalanceActivityInteraction"] = (
        clean_df["Balance"] * clean_df["IsActiveMember"]
    )
    clean_df["ZeroBalance"] = (clean_df["Balance"] == 0).astype("float64")
    clean_df["UnderUtilized"] = (clean_df["NumOfProducts"] == 1).astype("float64")
    clean_df["AgeRisk"] = ((clean_df["Age"] >= 50) & (clean_df["Age"] <= 65)).astype(
        "float64"
    )
    clean_df["GermanyRisk"] = (clean_df["Geography"] == "Germany").astype("float64")
    clean_df["GermanyMatureCombo"] = (
        clean_df["GermanyRisk"] * clean_df["AgeRisk"]
    ).astype("float64")

    # Remove geography column
    clean_df.drop("Geography", axis=1, inplace=True)

    X = clean_df.drop("Exited", axis=1)
    y = clean_df["Exited"]

    print(f"✅ Data loaded: {len(clean_df)} records, {len(X.columns)} features")
    return X, y


@task
def prepare_training_data(X, y):
    """Split data and create preprocessor."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=1
    )

    numerical_features = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "EstimatedSalary",
        "BalanceActivityInteraction",
    ]
    binary_features = [
        "HasCrCard",
        "IsActiveMember",
        "ZeroBalance",
        "UnderUtilized",
        "AgeRisk",
        "GermanyRisk",
        "GermanyMatureCombo",
    ]
    categorical_features = ["Gender"]

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numerical_features),
            ("bin", "passthrough", binary_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                categorical_features,
            ),
        ]
    )

    class_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    print(f"✅ Training split: {len(X_train)} train, {len(X_test)} test")
    print(
        f"   Features: {len(numerical_features)} numerical, {len(binary_features)} binary, {len(categorical_features)} categorical"
    )
    print(f"   Class ratio: {class_ratio:.2f}")
    return X_train, X_test, y_train, y_test, preprocessor, class_ratio


def train_model_generic(
    name, estimator, params, preprocessor, X_train, y_train, X_test, y_test
):
    """Generic model training function to eliminate code duplication."""
    pipeline = Pipeline([("preprocessor", preprocessor), ("clf", estimator)])

    gs = GridSearchCV(pipeline, params, cv=3, scoring="recall", n_jobs=-1, verbose=0)

    with mlflow.start_run(run_name=f"{name}_recall_optimized"):
        gs.fit(X_train, y_train)

        model = gs.best_estimator_
        y_pred = model.predict(X_test)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_params(gs.best_params_)
        mlflow.log_metrics(
            {
                "cv_recall": gs.best_score_,
                "test_recall": recall,
                "test_precision": precision,
                "test_f1": f1,
            }
        )

        signature = infer_signature(X_train, y_train)
        mlflow.sklearn.log_model(
            model, name="model", signature=signature, input_example=X_train.head(1)
        )

        # Business impact metrics
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        total_churners = (y_test == 1).sum()

        print(f"   {name}: CV Recall={gs.best_score_:.3f}, Test Recall={recall:.3f}")
        print(f"   → Caught {tp}/{total_churners} churners, Missed {fn} churners")

        return {
            "model": model,
            "cv_score": gs.best_score_,
            "test_recall": recall,
            "test_f1": f1,
            "test_precision": precision,
            "run_id": mlflow.active_run().info.run_id,
        }


@task
def train_all_models(preprocessor, class_ratio, X_train, y_train, X_test, y_test):
    """Train all models using consolidated parameters."""
    print("✅ Training models...")

    # Model configs
    models_config = {
        "LogisticRegression": {
            "estimator": LogisticRegression(random_state=42, max_iter=1000),
            "params": {
                "clf__C": [0.1, 1, 10],
                "clf__class_weight": [{0: 1, 1: 5}, {0: 1, 1: 6}],
            },
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
            "params": {
                "clf__n_estimators": [200, 300],
                "clf__max_depth": [None],
                "clf__min_samples_leaf": [1, 2],
                "clf__class_weight": [{0: 1, 1: 4}, {0: 1, 1: 5}],
            },
        },
        "XGBoost": {
            "estimator": XGBClassifier(eval_metric="logloss", random_state=42),
            "params": {
                "clf__n_estimators": [200, 300],
                "clf__max_depth": [8, 10],
                "clf__min_child_weight": [1],
                "clf__scale_pos_weight": [class_ratio * 2, class_ratio * 3],
            },
        },
        "LightGBM": {
            "estimator": LGBMClassifier(random_state=42, verbose=-1),
            "params": {
                "clf__n_estimators": [200, 300],
                "clf__max_depth": [8, 10],
                "clf__min_child_samples": [5, 10],
                "clf__class_weight": [{0: 1, 1: 5}, {0: 1, 1: 6}],
            },
        },
    }

    results = {}
    for name, config in models_config.items():
        results[name] = train_model_generic(
            name,
            config["estimator"],
            config["params"],
            preprocessor,
            X_train,
            y_train,
            X_test,
            y_test,
        )

    return results


@task
def analyze_and_register_best_model(results, X, y, model_name="bank-churn-classifier"):
    """Analyze features and register best model by recall."""

    # Print model comparison
    print("\n📊 Model Comparison (Recall-Focused):")
    for name, result in results.items():
        print(
            f"   {name}: Recall={result['test_recall']:.3f}, F1={result['test_f1']:.3f}"
        )

    # Feature correlation analysis
    correlations = {}
    for col in X.columns:
        if col not in ["Gender"]:  # Skip categorical
            correlations[col] = X[col].corr(y)

    # Select best model by recall
    best_name = max(results.keys(), key=lambda x: results[x]["test_recall"])
    best_result = results[best_name]

    # Register model
    client = MlflowClient()
    model_uri = f"runs:/{best_result['run_id']}/model"

    try:
        try:
            client.create_registered_model(model_name)
        except Exception:
            pass

        mv = client.create_model_version(
            name=model_name, source=model_uri, run_id=best_result["run_id"]
        )

        client.set_registered_model_alias(
            name=model_name, alias="production", version=mv.version
        )

        print(f"✅ Best model: {best_name}")
        print(f"   → CV Recall: {best_result['cv_score']:.3f}")
        print(f"   → Test Recall: {best_result['test_recall']:.3f}")
        print(f"   → Registered as v{mv.version}")

        return (
            best_name,
            best_result["model"],
            f"models:/{model_name}@production",
            correlations,
        )

    except Exception as e:
        print(f"❌ Registration failed: {e}")
        return best_name, best_result["model"], model_uri, correlations


@task
def save_reference_data_for_monitoring(X, y):
    """Save reference data for drift monitoring."""
    os.makedirs("monitoring", exist_ok=True)
    reference_data = X.copy()
    reference_data["Exited"] = y
    reference_data.to_parquet("monitoring/reference_data.parquet", index=False)
    print(f"✅ Reference data saved: {len(reference_data)} records for monitoring")
    return len(reference_data)


@flow
def churn_prediction_pipeline(df):
    """Churn prediction pipeline."""
    print("=" * 60)

    print("\n📋 STEP 1: Initialize MLflow")
    setup_mlflow()

    print("\n📋 STEP 2: Load and prepare data")
    X, y = load_and_prepare_data(df)

    print("\n📋 STEP 3: Prepare training data")
    X_train, X_test, y_train, y_test, preprocessor, class_ratio = prepare_training_data(
        X, y
    )

    print("\n📋 STEP 4: Train all models")
    results = train_all_models(
        preprocessor, class_ratio, X_train, y_train, X_test, y_test
    )

    print("\n📋 STEP 5: Select and register best model")
    best_name, best_model, model_uri, correlations = analyze_and_register_best_model(
        results, X, y
    )

    print("\n📋 STEP 6: Save reference data for monitoring")
    save_reference_data_for_monitoring(X, y)

    print("=" * 60)
    print(f"   Best Model: {best_name}")
    print(f"   Model URI: {model_uri}")

    return best_name, best_model, model_uri


def make_prediction(model, customer_data):
    """Make churn prediction with recall-optimized threshold."""
    df = pd.DataFrame([customer_data])
    probability = model.predict_proba(df)[0][1]
    prediction = 1 if probability > 0.3 else 0

    return {
        "prediction": int(prediction),
        "churn_probability": round(float(probability), 4),
        "risk_level": "High"
        if probability > 0.6
        else "Medium"
        if probability > 0.3
        else "Low",
    }


def main():
    df = pd.read_csv("data/raw/Churn_Modelling.csv")
    best_name, best_model, model_uri = churn_prediction_pipeline(df)

    # Test prediction
    test_customer = {
        "CreditScore": 600,
        "Gender": "Female",
        "Age": 55,
        "Tenure": 2,
        "Balance": 0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 40000,
        "BalanceActivityInteraction": 0,
        "ZeroBalance": 1,
        "UnderUtilized": 1,
        "AgeRisk": 1,
        "GermanyRisk": 1,
        "GermanyMatureCombo": 1,
    }

    result = make_prediction(best_model, test_customer)
    print(f"\nTest prediction: {result}")


if __name__ == "__main__":
    main()
