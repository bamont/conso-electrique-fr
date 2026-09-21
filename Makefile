DATASET ?= data/processed/dataset_30min.parquet
METEO_FC ?= data/raw/meteo_forecast_openmeteo.parquet
MODEL ?= models/prod

.PHONY: install test lint train api docker-build docker-run

install:
	poetry install --with dev,notebooks

test:
	poetry run pytest -q

lint:
	poetry run ruff check src tests

train:
	poetry run python -m conso.train --dataset $(DATASET) --meteo-fc $(METEO_FC) --out $(MODEL)

api:
	CONSO_MODEL=$(MODEL) CONSO_DATASET=$(DATASET) CONSO_METEO_FC=$(METEO_FC) poetry run uvicorn conso.api.main:app --reload

docker-build:
	docker build -t conso-api .

docker-run:
	docker run --rm -p 8000:8000 -v $(PWD)/data:/app/data \
		-e CONSO_DATASET=/app/$(DATASET) -e CONSO_METEO_FC=/app/$(METEO_FC) conso-api
