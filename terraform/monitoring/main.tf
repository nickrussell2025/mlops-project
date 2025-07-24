locals {
    infra = data.terraform_remote_state.infrastructure.outputs
}

# Create dedicated Grafana database
resource "google_sql_database" "grafana" {
    name     = "grafana"
    instance = "mlflow-db"
}

# Create read-only user for data sources
resource "google_sql_user" "grafana_readonly" {
    name     = "grafana_readonly"
    instance = "mlflow-db"
    password = var.grafana_readonly_password
}

resource "google_cloud_run_service" "grafana" {
    name     = "grafana-monitoring"
    location = local.infra.region

    template {
        metadata {
            annotations = {
                "autoscaling.knative.dev/maxScale"         = "5"
                "run.googleapis.com/execution-environment" = "gen2"
            }
        }
        spec {
            service_account_name = local.infra.service_account_email
            containers {
                image = "grafana/grafana:latest"
                
                resources {
                    limits = {
                        memory = "512Mi"
                        cpu    = "1000m"
                    }
                }
                
                # Security and basic config
                env {
                    name  = "GF_SERVER_HTTP_PORT"
                    value = "8080"
                }
                env {
                    name  = "GF_SECURITY_ADMIN_PASSWORD"
                    value = var.db_password
                }
                env {
                    name  = "GF_SECURITY_DISABLE_GRAVATAR"
                    value = "true"
                }
                env {
                    name  = "GF_SECURITY_COOKIE_SECURE"
                    value = "true"
                }
                
                # Use separate Grafana database for metadata
                env {
                    name  = "GF_DATABASE_TYPE"
                    value = "postgres"
                }
                env {
                    name  = "GF_DATABASE_HOST"
                    value = "${local.infra.cloud_sql_ip}:5432"
                }
                env {
                    name  = "GF_DATABASE_NAME"
                    value = google_sql_database.grafana.name
                }
                env {
                    name  = "GF_DATABASE_USER"
                    value = local.infra.database_user
                }
                env {
                    name  = "GF_DATABASE_PASSWORD"
                    value = var.db_password
                }
                env {
                    name  = "GF_DATABASE_SSL_MODE"
                    value = "disable"
                }
                
                ports {
                    container_port = 8080
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

resource "google_cloud_run_service_iam_member" "grafana_public" {
    service  = google_cloud_run_service.grafana.name
    location = google_cloud_run_service.grafana.location
    role     = "roles/run.invoker"
    member   = "allUsers"
}

output "grafana_url" {
    description = "Grafana monitoring dashboard URL"
    value       = google_cloud_run_service.grafana.status[0].url
}