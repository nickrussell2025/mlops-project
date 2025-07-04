MLFLOW_PORT := 5000
PREFECT_PORT := 4200
API_PORT := 8080
GRAFANA_PORT := 3000

.PHONY: help install test test-api test-cov test-integration test-fast lint format check-format fix clean run-api run-mlflow run-prefect run-pipeline

help:
	@echo "Available commands:"
	@echo "  install      - Install dependencies"
	@echo "  test         - Run tests"
	@echo "  test-api     - Run only API tests"
	@echo "  test-cov     - Run tests with coverage report"
	@echo "  test-fast    - Run tests excluding slow ones"
	@echo "  lint         - Run linting with ruff"
	@echo "  format       - Format code with ruff"
	@echo "  check-format - Check if code is formatted"
	@echo "  fix          - Auto-fix linting issues and format code"
	@echo "  check-all    - Run all quality checks (lint + format check + test)"
	@echo "  clean        - Clean build artifacts"
	@echo "  run-api      - Run Flask API"
	@echo "  run-mlflow   - Run MLflow UI"
	@echo "  run-pipeline - Run training pipeline"

install:
	uv sync --all-extras

test:
	uv run pytest tests/ -v

test-api:
	uv run pytest tests/test_api.py -v

test-cov:
	uv run pytest tests/ --cov=src --cov-report=html --cov-report=term

test-integration:
	uv run pytest tests/ -v -m integration

test-fast:
	uv run pytest tests/ -v -m "not slow"

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

check-format:
	uv run ruff format --check src tests

fix:
	uv run ruff check --fix src tests
	uv run ruff format src tests

check-all: lint check-format test

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage

run-api:
	uv run python -m src.mlops_churn.app --port $(API_PORT)

run-mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --port $(MLFLOW_PORT)

run-prefect:
	uv run prefect server start --port $(PREFECT_PORT)

run-pipeline:
	uv run python src/mlops_churn/pipeline.py