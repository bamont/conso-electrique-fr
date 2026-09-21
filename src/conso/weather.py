"""Météo : agrégation nationale, correction de biais horaire, accès Open-Meteo."""

from __future__ import annotations

import time

import pandas as pd
import requests

from .config import CITIES, CITY_WEIGHTS, HOURLY, OPENMETEO_FORECAST, TZ, WCOLS
from .timeutils import mask_between


def weighted_national(meteo: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    """Moyenne pondérée par la population, heure par heure.

    ``meteo`` : une ligne par (heure, ville), colonne ``city`` + variables Open-Meteo (``HOURLY``).
    Le poids d'une ville est ignoré pour les heures où sa valeur est manquante.
    """
    w_map = pd.Series(CITY_WEIGHTS if weights is None else weights)
    m = meteo.copy()
    w = m.pop("city").map(w_map)
    valid = m[HOURLY].notna().mul(w, axis=0)
    num = m[HOURLY].fillna(0).mul(w, axis=0).groupby(level=0).sum()
    den = valid.groupby(level=0).sum()
    out = (num / den).sort_index()
    return out.rename(columns=dict(zip(HOURLY, WCOLS, strict=True)))


def national_30min(nat_hourly: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Interpole la météo horaire au pas de 30 min et l'aligne sur ``index``."""
    return nat_hourly.resample("30min").interpolate(method="time").reindex(index)


def hourly_bias(wt_fc: pd.DataFrame, wt_obs: pd.DataFrame, start, end, min_per_hour: int = 100) -> pd.Series:
    """Biais moyen (prévu - observé) de la température, par heure locale, sur [start, end[."""
    ok = mask_between(wt_fc.index, start, end) & wt_fc["temp_nat"].notna().to_numpy() & wt_obs["temp_nat"].notna().to_numpy()
    if ok.sum() == 0:
        raise ValueError("Aucune paire prévu/observé sur la période demandée pour estimer le biais.")
    diff = (wt_fc["temp_nat"] - wt_obs["temp_nat"])[ok]
    hours = diff.index.tz_convert(TZ).hour
    grp = diff.groupby(hours)
    counts = grp.size().reindex(range(24), fill_value=0)
    if (counts < min_per_hour).any():
        raise ValueError(f"Pas assez de points pour estimer le biais horaire (min {counts.min()} < {min_per_hour}).")
    return grp.mean().reindex(range(24)).rename("biais_temp")


def apply_hourly_bias(wt: pd.DataFrame, bias: pd.Series) -> pd.DataFrame:
    """Retranche le biais horaire de la température (les autres variables sont inchangées)."""
    out = wt.copy()
    hours = pd.Series(out.index.tz_convert(TZ).hour, index=out.index)
    out["temp_nat"] = out["temp_nat"] - hours.map(bias).fillna(0.0)
    return out


def fetch_openmeteo(
    url: str, lat: float, lon: float, start: str | None = None, end: str | None = None,
    variables: list[str] | None = None, past_days: int | None = None, forecast_days: int | None = None,
    session=None, max_retry: int = 6,
) -> pd.DataFrame:
    """Télécharge des données horaires Open-Meteo (UTC). Gère la limite de débit (HTTP 429)."""
    variables = variables or HOURLY
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(variables), "timezone": "UTC"}
    if start and end:
        params.update(start_date=start, end_date=end)
    if past_days is not None:
        params["past_days"] = past_days
    if forecast_days is not None:
        params["forecast_days"] = forecast_days
    get = (session or requests).get
    for attempt in range(max_retry):
        r = get(url, params=params, timeout=90)
        if r.status_code == 200:
            df = pd.DataFrame(r.json()["hourly"])
            df["time"] = pd.to_datetime(df["time"], utc=True)
            return df.set_index("time")
        if r.status_code == 429:
            time.sleep(30 * (attempt + 1))
            continue
        r.raise_for_status()
    raise RuntimeError("Open-Meteo : trop de tentatives (limite de débit ?).")


def fetch_national_forecast(past_days: int = 45, forecast_days: int = 3, session=None) -> pd.DataFrame:
    """Météo nationale pondérée, horaire : ``past_days`` jours passés + ``forecast_days`` jours de prévision."""
    frames = []
    for city, (lat, lon, _) in CITIES.items():
        d = fetch_openmeteo(OPENMETEO_FORECAST, lat, lon, past_days=past_days, forecast_days=forecast_days, session=session)
        frames.append(d.assign(city=city))
    return weighted_national(pd.concat(frames).sort_index())
