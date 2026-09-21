"""Prévision d'un jour à partir d'un historique et d'une météo prévue."""

from __future__ import annotations

import pandas as pd

from .config import HISTORY_DAYS, ISSUE_HOUR, TZ
from .features import build_features_window
from .model import Bundle, predict_with_interval
from .timeutils import day_bounds
from .weather import apply_hourly_bias


def issue_time(target_date) -> pd.Timestamp:
    """Heure d'émission de la prévision du jour ``target_date`` : la veille à ``ISSUE_HOUR`` h (heure locale)."""
    d = pd.Timestamp(target_date).normalize()
    if d.tzinfo is not None:
        d = d.tz_localize(None)
    return (d - pd.Timedelta(days=1) + pd.Timedelta(hours=ISSUE_HOUR)).tz_localize(TZ)


def apply_issue_cutoff(d: pd.DataFrame, target_date) -> pd.DataFrame:
    """Efface la consommation à partir de l'heure d'émission (elle n'est pas connue à ce moment)."""
    out = d.copy()
    out.loc[out.index >= issue_time(target_date), "conso"] = float("nan")
    return out


def forecast_day(bundle: Bundle, d: pd.DataFrame, wt_raw: pd.DataFrame, target_date) -> pd.DataFrame:
    """Prévision (48 points en général) du jour ``target_date``.

    ``d``      : historique d'au moins ``HISTORY_DAYS`` jours + jour cible + jour suivant (calendrier),
                 avec la consommation connue jusqu'à l'émission (NaN ensuite) ;
    ``wt_raw`` : météo prévue brute (non débiaisée), même index que ``d``.
    """
    start, end = day_bounds(target_date)
    if d.index.min() > start - pd.Timedelta(days=HISTORY_DAYS - 1):
        raise ValueError(f"Historique insuffisant : il faut au moins {HISTORY_DAYS} jours avant le jour cible.")
    wt = apply_hourly_bias(wt_raw, bundle.bias)
    X = build_features_window(apply_issue_cutoff(d, target_date), wt, start, end)
    if X.empty:
        raise ValueError("Aucune ligne pour le jour cible.")
    return predict_with_interval(bundle, X)
