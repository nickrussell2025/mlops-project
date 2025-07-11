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

  metadata = {
    startup-script = <<-EOF
    #!/bin/bash
    set -e
    
    # Log all output
    exec > >(tee /var/log/startup-script.log) 2>&1
    echo "Starting PostgreSQL setup at $(date)"
    
    # Update packages
    apt-get update -y
    
    # Install PostgreSQL
    DEBIAN_FRONTEND=noninteractive apt-get install -y postgresql postgresql-contrib
    
    # Start PostgreSQL service
    systemctl start postgresql
    systemctl enable postgresql
    
    # Wait for PostgreSQL to be properly initialized after installation
    echo "Waiting 60 seconds for PostgreSQL to fully initialize..."
    sleep 60
    
    # Create database and set password
    sudo -u postgres createdb monitoring
    sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'test123';"
    
    # Configure PostgreSQL for network access
    PG_VERSION=$(sudo -u postgres psql -t -c "SELECT version();" | grep -oP '\d+\.\d+' | head -1)
    PG_CONFIG_DIR="/etc/postgresql/$PG_VERSION/main"
    
    # Backup original files
    cp $PG_CONFIG_DIR/postgresql.conf $PG_CONFIG_DIR/postgresql.conf.bak
    cp $PG_CONFIG_DIR/pg_hba.conf $PG_CONFIG_DIR/pg_hba.conf.bak
    
    # Configure listen addresses
    echo "listen_addresses = '*'" >> $PG_CONFIG_DIR/postgresql.conf
    
    # Configure host-based authentication
    echo "host all all 10.0.0.0/24 md5" >> $PG_CONFIG_DIR/pg_hba.conf
    
    # Restart PostgreSQL
    systemctl restart postgresql
    
    # Verify PostgreSQL is running and accepting connections
    sleep 5
    sudo -u postgres psql -c "SELECT 1;" -d monitoring
    
    echo "PostgreSQL setup completed successfully at $(date)"
  EOF
  }
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