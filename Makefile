COMPOSE := docker compose -f infra/docker-compose.yml

.DEFAULT_GOAL := help
.PHONY: help up down restart logs ps fmt lint type test check features train serve consume produce \
        up-app down-app label-sim monitor monitor-report

help:
	@echo "Targets:"
	@echo "  up       build and start the local stack, wait until healthy"
	@echo "  down     stop the stack and remove containers"
	@echo "  restart  down then up"
	@echo "  logs     follow logs for all services"
	@echo "  ps       show service status"
	@echo "  fmt      format code with ruff"
	@echo "  lint     lint code with ruff"
	@echo "  type     type check with mypy"
	@echo "  test     run the test suite"
	@echo "  check    lint, type check, and test"
	@echo "  features build offline features, then apply and materialize to Redis"
	@echo "  train    fit XGBoost + LightGBM with Optuna sweep and log to MLflow"
	@echo "  serve    serve the champion model over REST with BentoML"
	@echo "  consume  consume transactions, score them, publish predictions"
	@echo "  produce  replay transactions onto the stream at the configured rate"
	@echo "  up-app   build and start the serving and monitoring containers"
	@echo "  down-app stop the serving and monitoring containers"
	@echo "  label-sim   replay delayed ground-truth labels onto the stream"
	@echo "  monitor     run the drift and performance exporter"
	@echo "  monitor-report  build the Evidently drift report and log it to MLflow"

up:
	$(COMPOSE) up -d --build --wait

down:
	$(COMPOSE) down

restart: down up

logs:
	$(COMPOSE) logs -f

ps:
	$(COMPOSE) ps

fmt:
	uv run ruff format .

lint:
	uv run ruff check .

type:
	uv run mypy

test:
	uv run pytest

check: lint type test

features:
	PYTHONPATH=src uv run python -m fraud.features.build_offline
	PYTHONPATH=src uv run python -m fraud.features.materialize

train:
	PYTHONUNBUFFERED=1 PYTHONPATH=src uv run python -m fraud.training.train

serve:
	PYTHONPATH=src uv run bentoml serve src/fraud/serving/service.py:FraudService --port 3001

consume:
	PYTHONUNBUFFERED=1 PYTHONPATH=src uv run python -m fraud.ingestion.consumer

produce:
	PYTHONUNBUFFERED=1 PYTHONPATH=src uv run python -m fraud.ingestion.producer

up-app:
	$(COMPOSE) --profile app up -d --build --wait

down-app:
	$(COMPOSE) --profile app down

label-sim:
	PYTHONUNBUFFERED=1 PYTHONPATH=src uv run python -m fraud.ingestion.label_simulator

monitor:
	PYTHONUNBUFFERED=1 PYTHONPATH=src uv run python -m fraud.monitoring.exporter

monitor-report:
	PYTHONUNBUFFERED=1 PYTHONPATH=src uv run python -m pipelines.flows.monitoring_pipeline
