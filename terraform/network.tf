# terraform/network.tf

# VPC Network
resource "google_compute_network" "mlops_vpc" {
  name                    = "mlops-vpc"
  auto_create_subnetworks = false
  description            = "VPC for MLOps project"
}

# Subnet
resource "google_compute_subnetwork" "mlops_subnet" {
  name          = "mlops-subnet"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.mlops_vpc.id
  description   = "Subnet for MLOps VM"
}

# Firewall rule for Flask API
resource "google_compute_firewall" "allow_flask_api" {
  name    = "allow-flask-api"
  network = google_compute_network.mlops_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["8080"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["mlops-vm"]
  description   = "Allow Flask API access on port 8080"
}

# Firewall rule for SSH
resource "google_compute_firewall" "allow_ssh" {
  name    = "allow-ssh"
  network = google_compute_network.mlops_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["mlops-vm"]
  description   = "Allow SSH access"
}

# Firewall rule for MLflow (optional)
resource "google_compute_firewall" "allow_mlflow" {
  name    = "allow-mlflow"
  network = google_compute_network.mlops_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["5000"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["mlops-vm"]
  description   = "Allow MLflow UI access on port 5000"
}