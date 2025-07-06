# terraform/outputs.tf

output "vm_external_ip" {
  description = "External IP address of the VM"
  value       = google_compute_address.mlops_static_ip.address
}

output "vm_internal_ip" {
  description = "Internal IP address of the VM"
  value       = google_compute_instance.mlops_vm.network_interface[0].network_ip
}

output "vm_name" {
  description = "Name of the VM instance"
  value       = google_compute_instance.mlops_vm.name
}

output "ssh_command" {
  description = "SSH command to connect to the VM"
  value       = "ssh ubuntu@${google_compute_address.mlops_static_ip.address}"
}

output "flask_api_url" {
  description = "URL for the Flask API"
  value       = "http://${google_compute_address.mlops_static_ip.address}:8080"
}

output "mlflow_url" {
  description = "URL for MLflow UI"
  value       = "http://${google_compute_address.mlops_static_ip.address}:5000"
}