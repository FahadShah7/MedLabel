.PHONY: help setup data train test lint format check clean run

help:
	@echo "Available make targets:"
	@echo "  setup    - Create virtual environment and install dependencies"
	@echo "  data     - Download PubMedQA, validate, and split data"
	@echo "  train    - Train candidate models and log runs to MLflow"
	@echo "  test     - Run the pytest test suite"
	@echo "  lint     - Run ruff linter checks"
	@echo "  format   - Format code using ruff"
	@echo "  check    - Run lint and test suites"
	@echo "  clean    - Remove build and cache artifacts"
	@echo "  run      - Start the FastAPI development server"

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

data:
	.venv/bin/python src/prepare_data.py

train:
	.venv/bin/python src/train.py

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
