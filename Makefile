.PHONY: setup install data train run test format lint clean

# Default variables
PYTHON := python
UVICORN := uvicorn
HOST := 0.0.0.0
PORT := 8000

setup: install data train
	@echo "Setup complete!"

install:
	pip install -r requirements.txt

data:
	$(PYTHON) -m data.generate_data

train:
	$(PYTHON) -m training.train_model

run:
	$(UVICORN) app.main:app --host $(HOST) --port $(PORT) --reload

test:
	$(PYTHON) -m pytest tests/ -v

format:
	ruff format .

lint:
	ruff check .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
