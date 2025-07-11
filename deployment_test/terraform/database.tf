resource "google_compute_instance" "database_vm" {
  name         = "test-database"
  machine_type = "e2-micro"
  zone         = var.zone
  tags         = ["database"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
    }
  }

  network_interface {
    network    = google_compute_network.mlops_vpc.id
    subnetwork = google_compute_subnetwork.mlops_subnet.id
  }

  metadata_startup_script = <<-EOF
    apt-get update
    apt-get install -y postgresql
    sudo -u postgres createdb monitoring
    sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'test123';"
    echo "listen_addresses = '*'" >> /etc/postgresql/*/main/postgresql.conf
    echo "host all all 10.0.0.0/24 md5" >> /etc/postgresql/*/main/pg_hba.conf
    systemctl restart postgresql
  EOF
}

# Allow database access
resource "google_compute_firewall" "allow_postgres" {
  name    = "allow-postgres"
  network = google_compute_network.mlops_vpc.name
  allow {
    protocol = "tcp"
    ports    = ["5432"]
  }
  source_tags = ["mlops-vm"]
  target_tags = ["database"]
}