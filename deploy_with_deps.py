from prefect import flow

if __name__ == "__main__":
    flow.from_source(
        source="https://github.com/nickrussell2025/mlops-project.git",
        entrypoint="src/mlops_churn/churn_pipeline.py:churn_prediction_pipeline",
    ).deploy(
        name="churn-pipeline-cloud-run",
        work_pool_name="cloud-run-jobs",
        image="europe-west2-docker.pkg.dev/mlops-churn-prediction-465023/mlops-repo/training-pipeline:latest",
        job_variables={
            "env": {
                "MLFLOW_TRACKING_URI": "https://mlflow-working-139798376302.europe-west2.run.app",
                "DATABASE_URL": "postgresql://postgres:mlflow123secure@34.89.86.42:5432/monitoring"
            }
        }
    )