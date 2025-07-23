from prefect import flow

if __name__ == "__main__":
    flow.from_source(
        source="https://github.com/nickrussell2025/mlops-project.git",
        entrypoint="src/mlops_churn/churn_pipeline.py:churn_prediction_pipeline",
    ).deploy(
        name="churn-pipeline-v2",  # Changed name
        work_pool_name="my-managed-pool",
        job_variables={"pip_packages": ["mlflow", "pandas", "scikit-learn", "google-cloud-storage", "lightgbm", "xgboost", "imbalanced-learn"]}
    )