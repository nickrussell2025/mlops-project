# Database VM has hardcoded IP in subnet

# Data sources to reference existing network resources (don't manage them)
data "google_compute_network" "mlops_vpc" {
    name = "mlops-vpc"
}

data "google_compute_subnetwork" "mlops_subnet" {
    name = "mlops-subnet"
    region = var.region
}

# Static external IP
resource "google_compute_address" "mlops_static_ip" {
    name   = "mlops-static-ip"
    region = var.region
}

# Container-Optimized VM instance
resource "google_compute_instance" "mlops_vm" {
    name         = var.vm_name
    machine_type = var.machine_type
    zone         = var.zone

    tags = ["mlops-vm"]

    boot_disk {
        initialize_params {
            # Container-Optimized OS - includes Docker runtime
            image = "cos-cloud/cos-stable"
            size  = var.disk_size
            type  = "pd-ssd"
        }
    }

    network_interface {
        network    = data.google_compute_network.mlops_vpc.id
        subnetwork = data.google_compute_subnetwork.mlops_subnet.id

        access_config {
            nat_ip = google_compute_address.mlops_static_ip.address
        }
    }

    # Container configuration stored in metadata
    metadata = {
        # Container declaration for Container-Optimized OS
        gce-container-declaration = yamlencode({
            spec = {
                containers = [{
                    name  = "container-test-api"
                    image = "gcr.io/${var.project_id}/container-test-api:latest"
                    
                    ports = [{
                        containerPort = 8080
                    }]
                    
                    env = [
                        {
                            name  = "PORT"
                            value = "8080"
                        },
                        {
                            name  = "TZ"
                            value = "Europe/London"
                        },
                        {
                            name  = "DATABASE_URL"
                            value = "postgresql://postgres:test123@10.0.0.2:5432/monitoring"
                        }
                    ]
                    
                    # Resource limits for container
                    resources = {
                        limits = {
                            memory = "512Mi"
                            cpu    = "500m"
                        }
                        requests = {
                            memory = "256Mi"
                            cpu    = "250m"
                        }
                    }
                }]
                
                # Restart policy for container
                restartPolicy = "Always"
            }
        })
        
        # Enable Google Cloud monitoring
        google-monitoring-enabled = "true"
        google-logging-enabled    = "true"
    }

    # Service account with necessary permissions
    service_account {
        scopes = [
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/devstorage.read_only"
        ]
    }

    # Allow stopping for updates
    allow_stopping_for_update = true

    # Labels for resource management
    labels = {
        environment = "test"
        project     = "mlops-churn"
        container   = "cos-stable"
    }

    # Database VM is deployed separately
}