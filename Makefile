.PHONY: help install dev-install run-backend run-frontend run-all test lint format clean docker-build docker-run

# Variables
PYTHON := python3
PIP := pip3
BACKEND_PORT := 8000
FRONTEND_PORT := 8501

help:
	@echo "Titanic Chat Agent - Available Commands"
	@echo "========================================"
	@echo "make install       - Install production dependencies"
	@echo "make dev-install   - Install development dependencies"
	@echo "make run-backend   - Start the FastAPI backend"
	@echo "make run-frontend  - Start the Streamlit frontend"
	@echo "make run-all       - Start both backend and frontend"
	@echo "make test          - Run tests"
	@echo "make lint          - Run linting"
	@echo "make format        - Format code"
	@echo "make clean         - Clean up generated files"
	@echo "make docker-build  - Build Docker images"
	@echo "make docker-run    - Run with Docker Compose"

install:
	$(PIP) install -r requirements.txt

dev-install:
	$(PIP) install -r requirements-dev.txt
	pre-commit install

run-backend:
	$(PYTHON) -m uvicorn backend.main:app --host 0.0.0.0 --port $(BACKEND_PORT) --reload

run-frontend:
	streamlit run frontend/app.py --server.port $(FRONTEND_PORT)

run-all:
	@echo "Starting backend and frontend..."
	$(PYTHON) -m uvicorn backend.main:app --host 0.0.0.0 --port $(BACKEND_PORT) &
	sleep 3
	streamlit run frontend/app.py --server.port $(FRONTEND_PORT)

test:
	pytest tests/ -v --cov=backend --cov-report=term-missing

lint:
	flake8 backend/ frontend/ tests/
	mypy backend/ --ignore-missing-imports

format:
	black backend/ frontend/ tests/
	isort backend/ frontend/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ .coverage htmlcov/

docker-build:
	docker-compose build

docker-run:
	docker-compose up -d