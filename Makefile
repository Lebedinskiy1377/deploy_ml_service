COMPOSE ?= docker compose
PYTHON ?= python

.DEFAULT_GOAL := help
.PHONY: help up down logs seed train train-smoke demo clean install test lint format

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

up: ## Build and start PostgreSQL, MinIO, MLflow, the API and the UI
	$(COMPOSE) up -d --build

down: ## Stop the stack, keep the data volumes
	$(COMPOSE) down

logs: ## Follow the logs of all services
	$(COMPOSE) logs -f --tail=100

seed: ## Load the demo reference tables into PostgreSQL
	$(COMPOSE) run --rm seed

train: ## Train and register the demand model, then reload it in the API
	$(COMPOSE) run --rm --build trainer
	$(COMPOSE) restart api

train-smoke: ## Quick training run that checks the pipeline end to end
	$(COMPOSE) run --rm --build trainer --n-trials 1 --cv-splits 3 --max-estimators 300
	$(COMPOSE) restart api

demo: up seed train-smoke ## Stack, demo data and a model in one go

clean: ## Stop the stack and delete its volumes
	$(COMPOSE) down -v

install: ## Install test and lint dependencies into the active virtualenv
	$(PYTHON) -m pip install -r requirements-dev.txt

test: ## Run all test suites
	$(PYTHON) -m pytest

lint: ## Check code style
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format: ## Fix code style
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .
