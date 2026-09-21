"""API de prévision.

    CONSO_MODEL=models/prod CONSO_DATASET=data/processed/dataset_30min.parquet \\
    CONSO_METEO_FC=data/raw/meteo_forecast_openmeteo.parquet uvicorn conso.api.main:app

- ``mode=replay`` : rejoue la prévision d'un jour passé (hors ligne), avec réel et prévision RTE J-1 pour comparer ;
- ``mode=live``   : prévision du lendemain à partir des données RTE et Open-Meteo en direct.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from .. import __version__
from ..config import TZ
from ..model import Bundle, load_bundle
from ..pipeline import forecast_day, issue_time
from ..providers import LiveProvider, ReplayProvider

log = logging.getLogger("conso.api")


class ForecastPoint(BaseModel):
    time: str                       # début du créneau de 30 min, heure locale (ISO 8601)
    forecast_mw: float
    lower_mw: float                 # borne basse de l'intervalle à 90 %
    upper_mw: float                 # borne haute de l'intervalle à 90 %
    actual_mw: float | None = None  # rejeu uniquement
    rte_j1_mw: float | None = None  # rejeu uniquement


class ForecastResponse(BaseModel):
    date: str
    mode: str
    issue_time: str
    model_version: str
    n_points: int
    points: list[ForecastPoint]


def _num(x) -> float | None:
    return None if x is None or pd.isna(x) else float(x)


def create_app(bundle: Bundle | None = None, replay: ReplayProvider | None = None, live=None) -> FastAPI:
    app = FastAPI(title="Prévision de la consommation électrique française", version=__version__)
    live = live if live is not None else LiveProvider()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_loaded": bundle is not None, "replay_available": replay is not None}

    @app.get("/model")
    def model_info() -> dict:
        if bundle is None:
            raise HTTPException(503, "Modèle non chargé (variable CONSO_MODEL).")
        meta = {k: v for k, v in bundle.meta.items() if k not in ("features", "params")}
        return {**meta, "n_features": len(bundle.feats), "bias_hourly_c": {int(h): round(float(v), 3) for h, v in bundle.bias.items()},
                "interval_widening_mw": {k: round(v) for k, v in bundle.qhat.items()}}

    @app.get("/forecast", response_model=ForecastResponse)
    def forecast(
        date: str = Query(..., description="Jour à prévoir, AAAA-MM-JJ (heure locale)", pattern=r"^\d{4}-\d{2}-\d{2}$"),
        mode: str = Query("replay", pattern="^(replay|live)$"),
    ) -> ForecastResponse:
        if bundle is None:
            raise HTTPException(503, "Modèle non chargé (variable CONSO_MODEL).")
        try:
            target = pd.Timestamp(date)
        except ValueError as e:
            raise HTTPException(422, f"Date invalide : {date}") from e
        provider = replay if mode == "replay" else live
        if provider is None:
            raise HTTPException(503, "Rejeu indisponible (variables CONSO_DATASET et CONSO_METEO_FC).")
        try:
            inp = provider.get_inputs(target, bundle.bias)
            fc = forecast_day(bundle, inp.d, inp.wt_raw, target)
        except LookupError as e:
            raise HTTPException(404, str(e)) from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        except Exception as e:  # source externe indisponible, etc.
            log.exception("Échec de la prévision")
            raise HTTPException(503, f"Source de données indisponible : {type(e).__name__}") from e

        fc = fc.dropna(subset=["forecast"])
        if fc.empty:
            raise HTTPException(404, "Aucune prévision disponible (météo manquante).")
        points = []
        for ts, row in fc.iterrows():
            points.append(ForecastPoint(
                time=ts.tz_convert(TZ).isoformat(), forecast_mw=round(float(row["forecast"]), 1),
                lower_mw=round(float(row["lower"]), 1), upper_mw=round(float(row["upper"]), 1),
                actual_mw=_num(inp.actual.get(ts)) if inp.actual is not None else None,
                rte_j1_mw=_num(inp.rte_j1.get(ts)) if inp.rte_j1 is not None else None,
            ))
        return ForecastResponse(
            date=str(target.date()), mode=mode, issue_time=issue_time(target).isoformat(),
            model_version=str(bundle.meta.get("version", "inconnue")), n_points=len(points), points=points,
        )

    return app


def create_app_from_env() -> FastAPI:
    model_dir = os.environ.get("CONSO_MODEL")
    bundle = load_bundle(model_dir) if model_dir and Path(model_dir, "meta.json").exists() else None
    replay = None
    ds, fc = os.environ.get("CONSO_DATASET"), os.environ.get("CONSO_METEO_FC")
    if ds and fc and Path(ds).exists() and Path(fc).exists():
        replay = ReplayProvider.from_files(ds, fc)
    return create_app(bundle, replay)


app = create_app_from_env()
