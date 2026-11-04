.PHONY: setup data train test lint run

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

run:
	.venv/bin/uvicorn src.app:app --reload
