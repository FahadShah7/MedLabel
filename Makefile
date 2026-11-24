.PHONY: help setup data train eval mlflow test lint format check clean run docker-build docker-run

IMAGE ?= medlabel

help:
	@echo "Available make targets:"
	@echo "  setup        - Create virtual environment and install dependencies"
	@echo "  data         - Download PubMedQA, validate, and split data"
	@echo "  train        - Train candidate models and log runs to MLflow"
	@echo "  eval         - Evaluate Gemini prompts, models, parameters and temperature in MLflow"
	@echo "  mlflow       - Open the MLflow UI at http://127.0.0.1:5000"
	@echo "  test         - Run the pytest test suite"
	@echo "  lint         - Run ruff linter checks"
	@echo "  format       - Format code using ruff"
	@echo "  check        - Run lint and test suites"
	@echo "  clean        - Remove build and cache artifacts"
	@echo "  run          - Start the FastAPI development server"
	@echo "  docker-build - Build the Docker image"
	@echo "  docker-run   - Run the Docker image on port 8000"

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements-dev.txt

data:
	.venv/bin/python src/prepare_data.py

train:
	.venv/bin/python src/train.py

eval:
	.venv/bin/python -m src.evaluate_llm

mlflow:
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

test:
	.venv/bin/python -m pytest -v

lint:
	.venv/bin/ruff check .

format:
	.venv/bin/ruff format .

check: lint test

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ src/__pycache__ tests/__pycache__

run:
	.venv/bin/uvicorn src.app:app --reload

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 -e GEMINI_API_KEY $(IMAGE)
