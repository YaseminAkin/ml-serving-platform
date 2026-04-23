.PHONY: install install-dev lint format typecheck test test-cov up down clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .

format:
	black .
	ruff check --fix .

typecheck:
	mypy gateway registry eval

test:
	pytest tests/

test-cov:
	pytest tests/ --cov=gateway --cov=registry --cov=eval --cov-report=term-missing

up:
	docker compose -f infra/docker-compose.yml up -d

down:
	docker compose -f infra/docker-compose.yml down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist *.egg-info
