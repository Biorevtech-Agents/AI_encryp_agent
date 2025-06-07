.PHONY: install test lint format clean

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -e .

# Run tests
test:
	pytest tests/

# Run linting
lint:
	flake8 src/ tests/
	mypy src/ tests/

# Format code
format:
	black src/ tests/

# Clean up
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name "*.log" -delete
	find . -type f -name "agent_state.json" -delete

# Run the agent
run:
	python src/main.py

# Run in backtest mode
backtest:
	python src/main.py --backtest

# Create virtual environment
venv:
	python -m venv venv
	@echo "Run 'source venv/bin/activate' to activate the virtual environment" 