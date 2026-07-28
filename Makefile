.PHONY: install dev lint format typecheck test coverage clean

# Installation
install:
	pip install -r backend/requirements.txt
	pip install -r ai-engine/requirements.txt
	pip install -e ".[dev]"

dev:
	uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Quality
lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy backend/ ai-engine/

# Testing
test:
	pytest

coverage:
	pytest --cov=. --cov-report=html

# Docker
docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov dist build
