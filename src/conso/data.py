"""Chargement des jeux de données produits par les notebooks 01 et 03."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .weather import national_30min, weighted_national

NUMERIC_CALENDAR = ["vac_A", "vac_B", "vac_C", "n_zones_vac"]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Charge ``dataset_30min.parquet`` (index UTC, pas de 30 min)."""
    df = pd.read_parquet(path)
    for c in NUMERIC_CALENDAR:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_national_forecast(path: str | Path, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Charge les prévisions météo historiques par ville et les agrège au niveau national (30 min)."""
    meteo = pd.read_parquet(path)
    return national_30min(weighted_national(meteo), index)


def consolidated_end(df: pd.DataFrame) -> pd.Timestamp:
    """Borne exclusive de la période consolidée (dernier point `cons_def` + 30 min)."""
    if "source" in df.columns and df["source"].eq("cons_def").any():
        return df.index[df["source"].eq("cons_def")].max() + pd.Timedelta("30min")
    return df["conso"].last_valid_index() + pd.Timedelta("30min")
