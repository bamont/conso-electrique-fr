"""Métriques d'évaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def metrics(y_true: pd.Series, y_pred: pd.Series, scale: float) -> dict:
    """MAE, RMSE, MAPE, biais (prévu - réel) et MASE, sur les points où la prévision existe."""
    e = (y_pred - y_true).dropna()
    yt = y_true.loc[e.index]
    return {
        "MAE": float(e.abs().mean()),
        "RMSE": float(np.sqrt((e ** 2).mean())),
        "MAPE %": float((e.abs() / yt).mean() * 100),
        "biais": float(e.mean()),
        "MASE": float(e.abs().mean() / scale),
        "n": int(len(e)),
    }


def mase_scale(y: pd.Series, y_lag: pd.Series, mask: np.ndarray) -> float:
    """Dénominateur de la MASE : MAE du naïf saisonnier (retard de 7 jours) sur une période."""
    return float((y - y_lag).abs()[mask].mean())
