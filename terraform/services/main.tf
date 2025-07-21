# mlflow/main.tf - MLflow service with Bitnami image (Docker Hub)

locals {
    infra = data.terraform_remote_state.infrastructure.outputs
}

resource "google_cloud_run_service" "mlflow" {
    name     = "mlflow-working"
    location = local.infra.region

    template {
        metadata {
            annotations = {
                "autoscaling.knative.dev/maxScale"         = "10"
                "run.googleapis.com/execution-environment" = "gen2"
            }
        }
        spec {
            service_account_name = local.infra.service_account_email
            containers {
                # Use Bitnami MLflow image from Docker Hub (Cloud Run compatible)
                image = "bitnami/mlflow:latest"
                
                # Use shell command to properly handle Cloud Run PORT variable
                command = ["/bin/sh"]
                args = [
                    "-c",
                    "mlflow server --host 0.0.0.0 --port $PORT --backend-store-uri \"$MLFLOW_BACKEND_STORE_URI\" --default-artifact-root \"$MLFLOW_ARTIFACT_ROOT\""
                ]
                
                env {
                    name  = "MLFLOW_BACKEND_STORE_URI"
                    value = "postgresql://${local.infra.database_user}:${var.db_password}@${local.infra.cloud_sql_ip}:5432/${local.infra.database_name}"
                }
                env {
                    name  = "MLFLOW_ARTIFACT_ROOT"
                    value = local.infra.artifacts_bucket_url
                }
                
                resources {
                    limits = {
                        memory = "1Gi"
                        cpu    = "1000m"
                    }
                }
                
                ports {
                    container_port = 5000
                }
            }
            timeout_seconds = 300
        }
    }
    traffic {
        percent         = 100
        latest_revision = true
    }
}

resource "google_cloud_run_service_iam_member" "mlflow_public" {
    service  = google_cloud_run_service.mlflow.name
    location = google_cloud_run_service.mlflow.location
    role     = "roles/run.invoker"
    member   = "allUsers"
}

output "mlflow_url" {
    description = "MLflow tracking server URL"
    value       = google_cloud_run_service.mlflow.status[0].url
}