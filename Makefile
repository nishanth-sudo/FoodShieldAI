.PHONY: install dev lint format typecheck \
        test test-unit test-integration test-e2e \
        test-performance test-security test-robustness \
        coverage coverage-html clean

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
install:
	pip install -r backend/requirements.txt
	pip install -r aiengine/requirements.txt
	pip install -e ".[dev]"

dev:
	uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------
lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy backend/ aiengine/

# ---------------------------------------------------------------------------
# Testing — Phase 7
# ---------------------------------------------------------------------------

## Run ALL tests
test:
	pytest

## Unit tests only (fast, no external deps)
test-unit:
	pytest -m unit -v

## Integration tests (mocked DB/services)
test-integration:
	pytest -m integration -v

## End-to-end tests (full pipeline, mocked externals)
test-e2e:
	pytest -m e2e -v

## Performance / load tests
test-performance:
	pytest -m performance -v --tb=short

## Security audit tests
test-security:
	pytest -m "security or unit" tests/unit/test_security_audit.py -v

## AI model robustness tests
test-robustness:
	pytest -m robustness -v tests/unit/test_model_robustness.py

## AI engine unit tests
test-aiengine:
	pytest tests/unit/test_aiengine.py -v

## Coverage report (HTML)
coverage:
	pytest --cov=backend --cov=aiengine --cov-report=term-missing --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

coverage-html:
	python -m http.server 8080 --directory htmlcov

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov dist build
