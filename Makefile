.PHONY: help install install-dev test lint format check clean pre-commit

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	uv pip install -e .

install-dev: ## Install development dependencies
	uv pip install -e ".[dev]"
	pre-commit install

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=mcp_nvidia --cov-report=term --cov-report=html

lint: ## Run linter (Ruff)
	ruff check .

lint-fix: ## Run linter with auto-fix
	ruff check . --fix

format: ## Format code with Ruff
	ruff format .

format-check: ## Check code formatting
	ruff format --check .

typecheck: ## Run type checker (mypy)
	mypy src/

check: ## Run all checks (lint, format, type check)
	@echo "Running linter..."
	ruff check .
	@echo "\nChecking formatting..."
	ruff format --check .
	@echo "\nRunning type checker..."
	mypy src/
	@echo "\n✅ All checks passed!"

fix: ## Fix all auto-fixable issues
	ruff check . --fix
	ruff format .

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

clean: ## Clean up build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

sync-version: ## Sync version from pyproject.toml to package.json
	python scripts/sync-version.py

build: ## Build the package
	python -m build

publish-test: sync-version build ## Publish to TestPyPI
	python -m twine upload --repository testpypi dist/*

publish: sync-version build ## Publish to PyPI
	python -m twine upload dist/*

run: ## Run the MCP server (stdio mode)
	mcp-nvidia

run-http: ## Run the MCP server (HTTP mode)
	mcp-nvidia http

.DEFAULT_GOAL := help
