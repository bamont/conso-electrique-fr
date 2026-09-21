"""Client de l'API de prévision."""

from __future__ import annotations

import os

import pandas as pd
import requests

from ..config import TZ

DEFAULT_API_URL = "http://localhost:8000"


class ApiError(Exception):
    """Erreur renvoyée par l'API ou impossibilité de la joindre."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def now_paris() -> pd.Timestamp:
    """Heure actuelle à Paris."""
    return pd.Timestamp.now(tz=TZ)


def api_url() -> str:
    return os.environ.get("CONSO_API_URL", DEFAULT_API_URL).rstrip("/")


def get_json(path: str, params: dict | None = None, timeout: float = 120, base_url: str | None = None) -> dict:
    url = (base_url or api_url()) + path
    try:
        r = requests.get(url, params=params, timeout=timeout)
    except requests.RequestException as e:
        raise ApiError(f"API injoignable ({url}) : {type(e).__name__}") from e
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise ApiError(str(detail), r.status_code)
    return r.json()


def forecast_frame(payload: dict) -> pd.DataFrame:
    """Réponse de ``/forecast`` -> DataFrame indexé par l'heure locale (Europe/Paris)."""
    df = pd.DataFrame(payload["points"])
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(TZ)
    return df.set_index("time")
