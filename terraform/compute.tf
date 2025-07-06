# terraform/compute.tf

# Static external IP
resource "google_compute_address" "mlops_static_ip" {
  name   = "mlops-static-ip"
  region = var.region
}

# VM startup script
locals {
  startup_script = <<-EOF
    #!/bin/bash
    
    # Update system
    apt-get update
    apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
    
    # Install Docker
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # Install PostgreSQL
    apt-get install -y postgresql postgresql-contrib
    
    # Configure PostgreSQL
    systemctl start postgresql
    systemctl enable postgresql
    
    # Create database and user for MLOps
    sudo -u postgres psql <<PSQL
    CREATE DATABASE monitoring;
    CREATE USER postgres WITH PASSWORD 'example';
    GRANT ALL PRIVILEGES ON DATABASE monitoring TO postgres;
    ALTER USER postgres CREATEDB;
PSQL
    
    # Configure PostgreSQL to accept local connections
    echo "host all all 127.0.0.1/32 md5" >> /etc/postgresql/*/main/pg_hba.conf
    echo "listen_addresses = 'localhost'" >> /etc/postgresql/*/main/postgresql.conf
    systemctl restart postgresql
    
    # Create init.sql and run it
    cat > /tmp/init.sql <<SQL
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    input_data JSONB,
    prediction FLOAT,
    model_version VARCHAR(50)
);

CREATE TABLE drift_reports (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    drift_detected BOOLEAN,
    drift_score FLOAT,
    feature_name VARCHAR(100),
    drift_type VARCHAR(50),
    report_data JSONB
);
SQL
    
    # Run init.sql
    sudo -u postgres psql -d monitoring -f /tmp/init.sql
    
    # Add user to docker group
    usermod -aG docker $USER
    
    # Create app directory
    mkdir -p /opt/mlops-app
    chown $USER:$USER /opt/mlops-app
    
    # Log completion
    echo "VM setup completed at $(date)" >> /var/log/startup-script.log
  EOF
}

# VM instance
resource "google_compute_instance" "mlops_vm" {
  name         = var.vm_name
  machine_type = var.machine_type
  zone         = var.zone

  tags = ["mlops-vm"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = var.disk_size
      type  = "pd-ssd"
    }
  }

  network_interface {
    network    = google_compute_network.mlops_vpc.id
    subnetwork = google_compute_subnetwork.mlops_subnet.id

    access_config {
      nat_ip = google_compute_address.mlops_static_ip.address
    }
  }

  metadata = {
    ssh-keys = "ubuntu:${file("~/.ssh/id_rsa.pub")}"
  }

  metadata_startup_script = local.startup_script

  service_account {
    scopes = ["cloud-platform"]
  }

  allow_stopping_for_update = true
}