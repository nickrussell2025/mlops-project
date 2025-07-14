# model-api/main.tf - Model API service only

locals {
  infra  = data.terraform_remote_state.infrastructure.outputs
  mlflow = data.terraform_remote_state.mlflow.outputs
}

resource "google_cloud_run_service" "model_api" {
  name     = "model-api"
  location = local.infra.region

  template {
    metadata {
      annotations = {
        "autoscaling.knative.dev/maxScale" = "50"
        "run.googleapis.com/execution-environment" = "gen2"
      }
    }
    spec {
      service_account_name = local.infra.service_account_email
      containers {
        image = "gcr.io/mlops-churn-prediction-465023/model-api:latest"
        resources {
          limits = {
            memory = "1Gi"  # LESSON: 1GB for model loading
            cpu    = "1000m"
          }
        }
        env {
          name  = "MLFLOW_TRACKING_URI"
          value = local.mlflow.mlflow_url
        }
        env {
          name  = "DATABASE_HOST"
          value = local.infra.cloud_sql_ip
        }
        env {
          name  = "DATABASE_NAME"
          value = local.infra.database_name
        }
        env {
          name  = "DATABASE_USER"
          value = local.infra.database_user
        }
        env {
          name  = "DATABASE_PASSWORD"
          value = var.db_password
        }
        env {
          name  = "DEPLOYMENT_MODE"
          value = "cloud"
        }
        ports {
          container_port = 8080
        }
      }
      timeout_seconds = 300  # LESSON: Extended timeout for model loading
    }
  }
  traffic {
    percent         = 100
    latest_revision = true
  }
}

resource "google_cloud_run_service_iam_member" "model_api_public" {
  service  = google_cloud_run_service.model_api.name
  location = google_cloud_run_service.model_api.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

output "model_api_url" {
  description = "Model API service URL"
  value       = google_cloud_run_service.model_api.status[0].url
}