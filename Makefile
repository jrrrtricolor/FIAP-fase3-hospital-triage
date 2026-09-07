.PHONY: install check lint test prepare-data train mlflow-ui clean docker-build docker-run

POETRY ?= poetry
DOCKER_IMAGE ?= fiap-fase3-hospital-triage
DOCKER_TAG ?= $(shell awk -F'"' '/^version =/ {print $$2; exit}' pyproject.toml)
MLFLOW_PORT ?= 5001

install:
	$(POETRY) install

check: lint test
	$(POETRY) check

lint:
	$(POETRY) run ruff check src/hospital_triage tests

test:
	$(POETRY) run python -m unittest discover ./ml_prep_kit/tests
	$(POETRY) run python -m unittest discover ./tests

prepare-data:
	$(POETRY) run python -m hospital_triage.data_preparation

train:
	$(POETRY) run python -m hospital_triage.training

mlflow-ui:
	$(POETRY) run mlflow ui --backend-store-uri sqlite:///mldb/mlflow.db --port $(MLFLOW_PORT)

run-server:
	$(POETRY) run fastapi run src/hospital_triage/api.py

docker-build:
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run:
	docker run --rm -it \
		-v "$(PWD)/data:/app/data" \
		-v "$(PWD)/mlruns:/app/mlruns" \
		-v "$(PWD)/mldb:/app/mldb" \
		-v "$(PWD)/model:/app/model" \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
