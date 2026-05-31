# Convenience targets. On Windows, run the equivalent commands shown in the
# README if `make` is unavailable, or use `make` via Git Bash / WSL.
.PHONY: install train test lint api web docker

install:
	pip install -r requirements-dev.txt

train:
	PYTHONPATH=src python -m heart.train

test:
	PYTHONPATH=src pytest

lint:
	ruff check src api app tests

api:
	PYTHONPATH=src uvicorn api.main:app --reload

web:
	PYTHONPATH=src streamlit run app/streamlit_app.py

docker:
	docker compose up --build
