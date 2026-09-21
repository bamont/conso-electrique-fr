"""Performances du modèle de production face à RTE J-1, calculées à partir du backtest (notebook 04)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import TZ
from ..metrics import mase_scale
from ..timeutils import mask_between, shift_local

LABELS = {
    "rte_j1": "RTE J-1",
    "prod_oracle": "Modèle, météo observée (borne haute)",
    "prod_fc": "Modèle, météo prévue brute",
    "prod_fc_debiased": "Modèle, météo prévue débiaisée",
}
COLORS = {LABELS["rte_j1"]: "#d62728", LABELS["prod_oracle"]: "#7f7f7f",
          LABELS["prod_fc"]: "#ff7f0e", LABELS["prod_fc_debiased"]: "#1f77b4"}
TEMP_BINS = [-np.inf, 0, 5, 10, 15, 20, 25, np.inf]
TEMP_LABELS = ["< 0", "0-5", "5-10", "10-15", "15-20", "20-25", "> 25"]
DATASET_COLUMNS = ["conso", "prevision_j1", "temp_nat"]


def load(dataset_path: str | Path, backtest_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge le jeu de données (colonnes utiles) et les prédictions du backtest de production."""
    df = pd.read_parquet(dataset_path, columns=DATASET_COLUMNS)
    bt = pd.read_parquet(backtest_path)
    return df, bt


def eval_frame(df: pd.DataFrame, bt: pd.DataFrame) -> pd.DataFrame:
    """Réel, RTE J-1, prévisions du modèle et température, sur les horodatages communs à tous les modèles."""
    F = pd.DataFrame({"y": df["conso"], "rte_j1": df["prevision_j1"], "temp_nat": df["temp_nat"]}).join(
        bt.add_prefix("prod_"), how="inner")
    return F[["y", "temp_nat", *LABELS]].dropna()


def mase_denominator(df: pd.DataFrame) -> float:
    y = df["conso"]
    return mase_scale(y, shift_local(y, 7), mask_between(y.index, pd.Timestamp("2017-11-01", tz=TZ), pd.Timestamp("2025-01-01", tz=TZ)))


def summary(F: pd.DataFrame, scale: float | None = None) -> pd.DataFrame:
    """MAE, MAPE, biais, écart-type de l'erreur (et MASE si ``scale`` est fourni), une ligne par modèle."""
    rows = {}
    for col, label in LABELS.items():
        e = F[col] - F["y"]
        row = {"MAE (MW)": e.abs().mean(), "MAPE (%)": (e.abs() / F["y"]).mean() * 100, "Biais (MW)": e.mean(),
               "Écart-type (MW)": e.std()}
        if scale:
            row["MASE"] = e.abs().mean() / scale
        rows[label] = row
    return pd.DataFrame(rows).T


def _mae_by(F: pd.DataFrame, key) -> pd.DataFrame:
    out = {label: (F[col] - F["y"]).abs().groupby(key, observed=True).mean() for col, label in LABELS.items()}
    return pd.DataFrame(out)


def mae_by_month(F: pd.DataFrame) -> pd.DataFrame:
    return _mae_by(F, F.index.tz_convert(TZ).strftime("%Y-%m"))


def mae_by_temperature(F: pd.DataFrame) -> pd.DataFrame:
    return _mae_by(F, pd.cut(F["temp_nat"], TEMP_BINS, labels=TEMP_LABELS))


def mae_by_slot(F: pd.DataFrame) -> pd.DataFrame:
    hours = F.index.tz_convert(TZ).hour
    return _mae_by(F, pd.Series(np.where(hours < 10, "0 h - 9 h 30", "10 h - 23 h 30"), index=F.index))


def rte_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Erreur de la prévision RTE J-1 par année (MAPE et biais), sur tout l'historique."""
    D = pd.DataFrame({"y": df["conso"], "rte": df["prevision_j1"]}).dropna()
    e = D["rte"] - D["y"]
    year = D.index.tz_convert(TZ).year
    return pd.DataFrame({"MAPE RTE J-1 (%)": (e.abs() / D["y"] * 100).groupby(year).mean(),
                         "Biais RTE J-1 (MW)": e.groupby(year).mean()})
