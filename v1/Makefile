.PHONY: install install-dev test lint format clean build help

PYTHON := python3
PIP := pip3

help:
	@echo "SVA Toolkit - Development Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install      Install package"
	@echo "  install-dev  Install package with dev dependencies"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build package"
	@echo "  check-tools  Check for required external tools"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=sva_toolkit --cov-report=term-missing --cov-report=html

lint:
	@echo "Running ruff..."
	-ruff check src/
	@echo "Running mypy..."
	-mypy src/sva_toolkit --ignore-missing-imports

format:
	@echo "Formatting with ruff..."
	-ruff format src/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

build: clean
	$(PYTHON) -m build

check-tools:
	@echo "Checking for required external tools..."
	@echo ""
	@echo "Verible (required for AST parsing):"
	@which verible-verilog-syntax 2>/dev/null && echo "  ✓ Found" || echo "  ✗ Not found - install from https://github.com/chipsalliance/verible"
	@echo ""
	@echo "SymbiYosys (required for implication checking):"
	@which sby 2>/dev/null && echo "  ✓ Found" || echo "  ✗ Not found - install from https://github.com/YosysHQ/sby"
	@echo ""
	@echo "Yosys (required by SymbiYosys):"
	@which yosys 2>/dev/null && echo "  ✓ Found" || echo "  ✗ Not found - install from https://github.com/YosysHQ/yosys"
	@echo ""

# Development shortcuts
.PHONY: dev run-example

dev: install-dev
	@echo "Development environment ready!"

run-example:
	@echo "Running example: Parse sample SVA..."
	sva-ast parse "property ReqAck; @(posedge clk) req |-> ##[1:3] ack; endproperty"
	@echo ""
	@echo "Running example: Generate CoT..."
	sva-cot build "property ReqAck; @(posedge clk) req |-> ##[1:3] ack; endproperty"
