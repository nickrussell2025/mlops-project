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

        access_config {
            # Ephemeral external IP for internet access
        }
    }

    metadata = {
        startup-script = <<-EOF
        #!/bin/bash
        set -e
        
        # Log all output
        exec > >(tee /var/log/startup-script.log) 2>&1
        echo "Starting PostgreSQL setup at $(date)"
        
        # Wait for network initialization
        sleep 30
        
        # Test network connectivity
        ping -c 3 8.8.8.8 || { echo "No internet connectivity"; exit 1; }
        
        # Update packages with retry
        for i in {1..3}; do
            apt-get update -y && break
            echo "Update attempt $i failed, retrying in 15 seconds..."
            sleep 15
        done
        
        # Install PostgreSQL
        DEBIAN_FRONTEND=noninteractive apt-get install -y postgresql postgresql-contrib
        
        # Start PostgreSQL service
        systemctl start postgresql
        systemctl enable postgresql
        
        # Wait for PostgreSQL to be ready
        echo "Waiting 30 seconds for PostgreSQL to initialize..."
        sleep 30
        
        # Create database and set password
        sudo -u postgres createdb monitoring
        sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'test123';"
        
        # Configure PostgreSQL using wildcards (no version detection needed)
        echo "listen_addresses = '*'" >> /etc/postgresql/*/main/postgresql.conf
        echo "host all all 10.0.0.0/24 md5" >> /etc/postgresql/*/main/pg_hba.conf
        
        # Restart PostgreSQL
        systemctl restart postgresql
        
        # Wait and verify
        sleep 10
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