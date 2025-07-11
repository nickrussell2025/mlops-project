# MLOps Bank Churn Prediction - Complete Project Makefile
# Comprehensive commands for development, testing, deployment, and monitoring

# Configuration
PROJECT_ID := mlops-churn-prediction-465023
CONTAINER_NAME := mlops-churn-api
API_PORT := 8080
MLFLOW_PORT := 5000
PREFECT_PORT := 4200
GRAFANA_PORT := 3000

.PHONY: help install test lint format clean build deploy monitor teardown

# Default target
help:
	@echo "🚀 MLOps Bank Churn Prediction - Available Commands:"
	@echo ""
	@echo "📦 SETUP & DEVELOPMENT:"
	@echo "  install           - Install all dependencies"
	@echo "  install-dev       - Install with development dependencies"
	@echo "  setup-env         - Create .env file from template"
	@echo ""
	@echo "🧪 TESTING & QUALITY:"
	@echo "  test              - Run all tests"
	@echo "  test-api          - Run API tests only"
	@echo "  test-integration  - Run integration tests"
	@echo "  test-cov          - Run tests with coverage report"
	@echo "  lint              - Run code linting"
	@echo "  format            - Format code with ruff"
	@echo "  check-all         - Run all quality checks"
	@echo ""
	@echo "🏗️ BUILD & DEPLOYMENT:"
	@echo "  build             - Build Docker container"
	@echo "  push              - Push container to Google Container Registry"
	@echo "  deploy-infra      - Deploy infrastructure with Terraform"
	@echo "  deploy-app        - Deploy application container"
	@echo "  deploy            - Complete deployment (infra + app)"
	@echo ""
	@echo "🏃 LOCAL DEVELOPMENT:"
	@echo "  run-api           - Run Flask API locally"
	@echo "  run-mlflow        - Run MLflow UI"
	@echo "  run-prefect       - Run Prefect server"
	@echo "  run-pipeline      - Execute ML training pipeline"
	@echo "  run-monitoring    - Start monitoring stack (Grafana, PostgreSQL)"
	@echo ""
	@echo "📊 MONITORING & LOGS:"
	@echo "  logs              - View application logs"
	@echo "  logs-container    - View container logs from deployed VM"
	@echo "  monitor           - Open monitoring dashboard"
	@echo "  health-check      - Check API health"
	@echo ""
	@echo "🧹 CLEANUP:"
	@echo "  clean             - Clean build artifacts"
	@echo "  clean-data        - Clean generated data files"
	@echo "  teardown          - Destroy all infrastructure"
	@echo "  reset             - Complete reset (clean + teardown)"

# SETUP & DEVELOPMENT
install:
	@echo "📦 Installing dependencies..."
	uv sync --all-extras

install-dev:
	@echo "📦 Installing development dependencies..."
	uv sync --all-extras --dev

setup-env:
	@echo "⚙️ Setting up environment file..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✅ Created .env file from template"; \
		echo "🔧 Please update .env with your configuration"; \
	else \
		echo "⚠️ .env file already exists"; \
	fi

# TESTING & QUALITY
test:
	@echo "🧪 Running all tests..."
	uv run pytest tests/ -v -W ignore::DeprecationWarning

test-api:
	@echo "🧪 Running API tests..."
	uv run pytest tests/test_api.py -v

test-integration:
	@echo "🧪 Running integration tests..."
	uv run pytest tests/ -v -m integration

test-cov:
	@echo "🧪 Running tests with coverage..."
	uv run pytest tests/ --cov=src --cov-report=html --cov-report=term

lint:
	@echo "🔍 Running linting..."
	uv run ruff check src tests

format:
	@echo "✨ Formatting code..."
	uv run ruff format src tests

check-all: lint test
	@echo "✅ All quality checks completed"

# BUILD & DEPLOYMENT
build:
	@echo "🏗️ Building Docker container..."
	docker build -t gcr.io/$(PROJECT_ID)/$(CONTAINER_NAME):latest .
	@echo "✅ Container built successfully"

push: build
	@echo "📤 Pushing container to Google Container Registry..."
	gcloud auth configure-docker --quiet
	docker push gcr.io/$(PROJECT_ID)/$(CONTAINER_NAME):latest
	@echo "✅ Container pushed successfully"

deploy-infra:
	@echo "🏗️ Deploying infrastructure with Terraform..."
	cd terraform && \
	terraform init && \
	terraform plan && \
	terraform apply -auto-approve
	@echo "✅ Infrastructure deployed successfully"

deploy-app: push
	@echo "🚀 Deploying application..."
	@echo "Application will be automatically deployed via Container-Optimized OS"
	@echo "✅ Application deployment initiated"

deploy: deploy-infra deploy-app
	@echo "🎉 Complete deployment finished!"
	@echo "🌐 API URL: $$(cd terraform && terraform output -raw flask_api_url)"

# LOCAL DEVELOPMENT
run-api:
	@echo "🏃 Starting Flask API locally..."
	uv run python -m src.mlops_churn.app --port $(API_PORT)

run-mlflow:
	@echo "🏃 Starting MLflow UI..."
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --port $(MLFLOW_PORT)

run-prefect:
	@echo "🏃 Starting Prefect server..."
	uv run prefect server start --port $(PREFECT_PORT)

run-pipeline:
	@echo "🏃 Running ML training pipeline..."
	uv run python src/mlops_churn/pipeline.py

run-monitoring:
	@echo "🏃 Starting monitoring stack..."
	docker-compose up -d
	@echo "✅ Monitoring stack started"
	@echo "📊 Grafana: http://localhost:$(GRAFANA_PORT)"
	@echo "🗄️ Adminer: http://localhost:8081"

# MONITORING & LOGS
logs:
	@echo "📋 Viewing local application logs..."
	tail -f logs/*.log

logs-container:
	@echo "📋 Viewing container logs from deployed VM..."
	@VM_IP=$$(cd terraform && terraform output -raw vm_external_ip 2>/dev/null || echo "NOT_DEPLOYED"); \
	if [ "$$VM_IP" != "NOT_DEPLOYED" ]; then \
		echo "Connecting to VM: $$VM_IP"; \
		gcloud compute ssh mlops-churn-vm --zone=europe-west2-a --command="sudo docker logs \$$(sudo docker ps -q --filter ancestor=gcr.io/$(PROJECT_ID)/$(CONTAINER_NAME):latest)"; \
	else \
		echo "❌ Infrastructure not deployed. Run 'make deploy-infra' first"; \
	fi

monitor:
	@echo "📊 Opening monitoring dashboard..."
	@VM_IP=$$(cd terraform && terraform output -raw vm_external_ip 2>/dev/null || echo "NOT_DEPLOYED"); \
	if [ "$$VM_IP" != "NOT_DEPLOYED" ]; then \
		echo "🌐 Opening http://$$VM_IP:$(GRAFANA_PORT)"; \
		open "http://$$VM_IP:$(GRAFANA_PORT)" || echo "Visit: http://$$VM_IP:$(GRAFANA_PORT)"; \
	else \
		echo "🌐 Opening local Grafana: http://localhost:$(GRAFANA_PORT)"; \
		open "http://localhost:$(GRAFANA_PORT)" || echo "Visit: http://localhost:$(GRAFANA_PORT)"; \
	fi

health-check:
	@echo "🏥 Checking API health..."
	@VM_IP=$$(cd terraform && terraform output -raw vm_external_ip 2>/dev/null || echo "localhost"); \
	PORT=$$(if [ "$$VM_IP" = "localhost" ]; then echo "$(API_PORT)"; else echo "$(API_PORT)"; fi); \
	echo "Checking health at: http://$$VM_IP:$$PORT/"; \
	curl -f "http://$$VM_IP:$$PORT/" || echo "❌ Health check failed"

# CLEANUP
clean:
	@echo "🧹 Cleaning build artifacts..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
	@echo "✅ Build artifacts cleaned"

clean-data:
	@echo "🧹 Cleaning generated data files..."
	rm -rf logs/*.log
	rm -rf mlruns/
	rm -rf mlartifacts/
	rm -rf monitoring/reports/
	@echo "✅ Data files cleaned"

teardown:
	@echo "🔥 Destroying infrastructure..."
	cd terraform && terraform destroy -auto-approve
	@echo "✅ Infrastructure destroyed"

reset: clean clean-data teardown
	@echo "🔄 Complete reset completed"

# UTILITY COMMANDS
check-env:
	@echo "🔍 Checking environment..."
	@command -v uv >/dev/null 2>&1 || { echo "❌ uv not installed"; exit 1; }
	@command -v docker >/dev/null 2>&1 || { echo "❌ Docker not installed"; exit 1; }
	@command -v gcloud >/dev/null 2>&1 || { echo "❌ Google Cloud CLI not installed"; exit 1; }
	@command -v terraform >/dev/null 2>&1 || { echo "❌ Terraform not installed"; exit 1; }
	@echo "✅ All required tools are installed"

init: check-env setup-env install
	@echo "🎉 Project initialization completed!"
	@echo "📚 Run 'make help' to see available commands"

# Advanced deployment commands
deploy-staging: 
	@echo "🚀 Deploying to staging environment..."
	@export TF_VAR_environment=staging && $(MAKE) deploy

deploy-production:
	@echo "🚀 Deploying to production environment..."
	@export TF_VAR_environment=production && $(MAKE) deploy

# CI/CD integration
ci-test: install test lint
	@echo "✅ CI tests completed successfully"

ci-deploy: ci-test build push deploy-infra
	@echo "✅ CI deployment completed successfully"