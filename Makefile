PYTHON ?= python
BACKEND := customer-service-app/backend
FRONTEND := customer-service-app/frontend
KNOWLEDGE := knowledge-pipeline
MODEL := model-pipeline

.PHONY: format lint test test-backend test-knowledge test-model frontend-check check

format:
	$(PYTHON) -m ruff format \
		$(BACKEND)/app $(BACKEND)/benchmarks $(BACKEND)/scripts $(BACKEND)/tests \
		$(KNOWLEDGE)/knowledge_pipeline $(KNOWLEDGE)/scripts $(KNOWLEDGE)/tests \
		$(MODEL)/model_pipeline $(MODEL)/scripts $(MODEL)/tests model-serving/smoke_test.py

lint:
	$(PYTHON) -m ruff check $(BACKEND) $(KNOWLEDGE) $(MODEL) model-serving/smoke_test.py
	bash -n model-serving/serve.sh

test-backend:
	cd $(BACKEND) && $(PYTHON) -m pytest tests

test-knowledge:
	cd $(KNOWLEDGE) && $(PYTHON) -m pytest tests

test-model:
	cd $(MODEL) && $(PYTHON) -m pytest tests

test: test-backend test-knowledge test-model

frontend-check:
	npm --prefix $(FRONTEND) test
	npm --prefix $(FRONTEND) run build

check: lint test frontend-check
