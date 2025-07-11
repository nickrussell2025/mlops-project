# terraform/variables.tf

variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region"
  type        = string
  default     = "europe-west2"
}

variable "zone" {
  description = "The GCP zone"
  type        = string
  default     = "europe-west2-a"
}

variable "machine_type" {
  description = "The VM machine type"
  type        = string
  default     = "e2-standard-2"
}

variable "vm_name" {
  description = "Name of the VM instance"
  type        = string
  default     = "mlops-churn-vm"
}

variable "disk_size" {
  description = "Boot disk size in GB"
  type        = number
  default     = 50
}