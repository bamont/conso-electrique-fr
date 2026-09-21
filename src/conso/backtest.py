"""Backtest en *rolling origin* avec le code de production (réentraînement mensuel)."""

from __future__ import annotations

import pandas as pd

from .config import GAP_DAYS, TZ, WCOLS
from .features import build_features, build_features_window
from .model import fit_lgb, predict_level
from .timeutils import mask_between
from .weather import apply_hourly_bias, hourly_bias


def rolling_backtest(
    df: pd.DataFrame, wt_fc: pd.DataFrame, test_start: pd.Timestamp, test_end: pd.Timestamp,
    train_start: str = "2013-01-01", rounds: int = 1000, params: dict | None = None,
    half_life: float | None = None, bias_months: int = 24, gap_days: int = GAP_DAYS,
    progress: bool = True, X_obs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Prévisions du test pour trois modes de météo, avec un modèle réentraîné chaque mois.

    Colonnes : ``oracle`` (météo observée), ``fc`` (prévue, brute), ``fc_debiased`` (prévue, température
    débiaisée par heure avec un biais réestimé chaque mois sur les ``bias_months`` mois précédents).
    Le modèle est entraîné sur la météo observée et n'utilise que des données antérieures au mois prédit
    moins ``gap_days`` jours.
    """
    y = df["conso"]
    wt_obs = df[WCOLS]
    X_obs = build_features(df, wt_obs) if X_obs is None else X_obs
    t0 = pd.Timestamp(train_start, tz=TZ)
    test_end = test_end.tz_convert(TZ)
    months = list(pd.date_range(test_start, test_end - pd.Timedelta("30min"), freq="MS"))
    bounds = months[1:] + [test_end]

    parts = []
    for ms, me in zip(months, bounds, strict=True):
        train_end = ms - pd.Timedelta(days=gap_days)
        model = fit_lgb(X_obs, y, mask_between(X_obs.index, t0, train_end), rounds, params, half_life, train_end)
        bias = hourly_bias(wt_fc, wt_obs, train_end - pd.DateOffset(months=bias_months), train_end)
        X_fc = build_features_window(df, wt_fc, ms, me)
        X_fcd = build_features_window(df, apply_hourly_bias(wt_fc, bias), ms, me)
        pm = mask_between(X_obs.index, ms, me)
        parts.append(pd.DataFrame({
            "oracle": predict_level(model, X_obs.loc[pm]),
            "fc": predict_level(model, X_fc),
            "fc_debiased": predict_level(model, X_fcd),
        }))
        if progress:
            print(f"  {ms.strftime('%Y-%m')} ok", end="")
    if progress:
        print()
    return pd.concat(parts)
