# MLOps Bank Churn Prediction - Makefile

# Configuration
PROJECT_ID := mlops-churn-prediction-465023
CONTAINER_NAME := mlops-churn-api
API_PORT := 8080
MLFLOW_PORT := 5000

.PHONY: help install test lint format fix clean build deploy

# Default target
help:
	@echo "MLOps Bank Churn Prediction - Available Commands:"
	@echo ""
	@echo "SETUP:"
	@echo "  install           - Install all dependencies"
	@echo "  setup-env         - Create .env file from template"
	@echo ""
	@echo "DEVELOPMENT:"
	@echo "  test              - Run all tests"
	@echo "  lint              - Check code quality"
	@echo "  format            - Format code"
	@echo "  fix               - Fix formatting and linting issues"
	@echo ""
	@echo "LOCAL SERVICES:"
	@echo "  run-api           - Run Flask API locally"
	@echo "  run-mlflow        - Run MLflow UI locally"
	@echo "  run-pipeline      - Execute ML training pipeline"
	@echo ""
	@echo "DEPLOYMENT:"
	@echo "  build             - Build Docker container"
	@echo "  push              - Push container to registry"
	@echo "  deploy-infra      - Deploy infrastructure (Terraform)"
	@echo "  deploy-mlflow     - Deploy MLflow service"
	@echo "  deploy-api        - Deploy model API"
	@echo ""
	@echo "CLEANUP:"
	@echo "  clean             - Clean build artifacts"
	@echo "  teardown          - Destroy all infrastructure"

# SETUP
install:
	uv sync --all-extras

setup-env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file from template"; \
	else \
		echo ".env file already exists"; \
	fi

# DEVELOPMENT
test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

fix:
	uv run ruff format src tests
	uv run ruff check src tests --fix

# LOCAL SERVICES
run-api:
	uv run python -m src.mlops_churn.app

run-mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port $(MLFLOW_PORT)

run-pipeline:
	uv run python src/mlops_churn/churn_pipeline.py

# DEPLOYMENT
build:
	docker build -f Dockerfile.model-api -t gcr.io/$(PROJECT_ID)/model-api:latest .

push: build
	gcloud auth configure-docker --quiet
	docker push gcr.io/$(PROJECT_ID)/model-api:latest

deploy-infra:
	cd terraform/infrastructure && terraform init && terraform apply

deploy-mlflow:
	cd terraform/services && terraform init && terraform apply

deploy-api:
	cd terraform/model-api && terraform init && terraform apply

# CLEANUP
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/

teardown:
	cd terraform/model-api && terraform destroy -auto-approve
	cd terraform/mlflow && terraform destroy -auto-approve  
	cd terraform/infrastructure && terraform destroy -auto-approve