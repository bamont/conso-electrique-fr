FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONPATH=/app/src \
    CONSO_MODEL=/app/models/prod

# LightGBM a besoin de libgomp
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry>=2,<3"

WORKDIR /app
COPY pyproject.toml poetry.lock ./
# uniquement les dépendances du groupe principal (pas de notebooks, pas de dev)
RUN poetry install --only main --no-root --no-interaction

COPY src ./src
COPY models ./models

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "conso.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
