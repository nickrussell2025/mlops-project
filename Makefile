.PHONY: help install test lint format check-format fix clean run-api run-mlflow run-pipeline

help:
	@echo "Available commands:"
	@echo "  install      - Install dependencies"
	@echo "  test         - Run tests"
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
	uv run python -m src.mlops_churn.app

run-mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

run-pipeline:
	uv run python src/mlops_churn/pipeline.py