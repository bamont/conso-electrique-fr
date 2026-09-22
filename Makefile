DATASET ?= data/processed/dataset_30min.parquet
METEO_FC ?= data/raw/meteo_forecast_openmeteo.parquet
MODEL ?= models/prod

.PHONY: install test lint train api dashboard journal-record journal-reconcile journal-summary up down docker-build docker-run

install:
	poetry install --with dev,notebooks,dashboard

test:
	poetry run pytest -q

lint:
	poetry run ruff check src tests

train:
	poetry run python -m conso.train --dataset $(DATASET) --meteo-fc $(METEO_FC) --out $(MODEL)

api:
	CONSO_MODEL=$(MODEL) CONSO_DATASET=$(DATASET) CONSO_METEO_FC=$(METEO_FC) poetry run uvicorn conso.api.main:app --reload

dashboard:
	CONSO_API_URL=http://localhost:8000 poetry run streamlit run src/conso/dashboard/app.py

journal-record:
	CONSO_MODEL=$(MODEL) poetry run python -m conso.journal record

journal-reconcile:
	poetry run python -m conso.journal reconcile

journal-summary:
	poetry run python -m conso.journal summary

up:
	docker compose up --build -d

down:
	docker compose down

docker-build:
	docker build -t conso-api .

docker-run:
	docker run --rm -p 8000:8000 -v $(PWD)/data:/app/data \
		-e CONSO_DATASET=/app/$(DATASET) -e CONSO_METEO_FC=/app/$(METEO_FC) conso-api
