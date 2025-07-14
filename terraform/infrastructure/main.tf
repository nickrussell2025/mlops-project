# infrastructure/main.tf - Stable foundation (database, storage, IAM)

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
  backend "gcs" {
    bucket = "mlops-churn-prediction-465023-terraform-state"
    prefix = "infrastructure/state"
  }
}

provider "google" {
  project = "mlops-churn-prediction-465023"
  region  = "europe-west2"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

# Cloud SQL PostgreSQL Instance
resource "google_sql_database_instance" "mlflow_db" {
  name             = "mlflow-db"
  database_version = "POSTGRES_14"
  region           = "europe-west2"
  
  settings {
    tier = "db-f1-micro"
    ip_configuration {
      ipv4_enabled = true
      authorized_networks {
        name  = "allow-all"
        value = "0.0.0.0/0"
      }
    }
    backup_configuration {
      enabled    = true
      start_time = "03:00"
    }
    disk_autoresize = true
    disk_size      = 20
    disk_type      = "PD_SSD"
  }
  deletion_protection = false
}

resource "google_sql_database" "mlflow" {
  name     = "mlflow"
  instance = google_sql_database_instance.mlflow_db.name
}

resource "google_sql_user" "mlflow_user" {
  name     = "postgres"
  instance = google_sql_database_instance.mlflow_db.name
  password = var.db_password
}

# GCS Bucket for MLflow Artifacts
resource "google_storage_bucket" "mlflow_artifacts" {
  name     = "mlops-churn-prediction-465023-mlflow-artifacts"
  location = "europe-west2"
  
  versioning {
    enabled = true
  }
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
  uniform_bucket_level_access = true
  force_destroy = false
}

# Service Account for Cloud Run Services
resource "google_service_account" "cloud_run_sa" {
  account_id   = "mlops-cloud-run-sa"
  display_name = "MLOps Cloud Run Service Account"
}

resource "google_storage_bucket_iam_member" "artifacts_access" {
  bucket = google_storage_bucket.mlflow_artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

resource "google_project_iam_member" "cloud_run_invoker" {
  project = "mlops-churn-prediction-465023"
  role    = "roles/run.invoker"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}