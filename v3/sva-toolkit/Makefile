.PHONY: install install-dev test lint format clean build

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

clean:
	rm -rf build dist .pytest_cache .ruff_cache .mypy_cache .coverage src/sva_toolkit.egg-info
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +

build:
	python -m build
