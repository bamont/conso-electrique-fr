"""Entraînement du modèle de production.

    python -m conso.train --dataset data/processed/dataset_30min.parquet \\
        --meteo-fc data/raw/meteo_forecast_openmeteo.parquet --out models/prod

Procédure :
1. Le modèle ponctuel est entraîné sur la météo **observée**, depuis ``train_start`` jusqu'à ``train_end``.
2. Les modèles quantiles (5 % / 95 %) sont entraînés avant la fenêtre de calibration ; les scores de
   non-conformité sont mesurés sur les ``calib_months`` derniers mois avec la **météo prévue débiaisée**
   (biais estimé uniquement avant cette fenêtre), puis regroupés par régime de température.
3. Le biais horaire de la température prévue de production est estimé sur les ``bias_months`` derniers mois.
"""

from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd

from .config import DEFAULT_PARAMS, DEFAULT_ROUNDS, FC_START, GAP_DAYS, QUANTILE_ROUNDS, TZ, WCOLS
from .data import consolidated_end, load_dataset, load_national_forecast
from .features import FEATURE_COLUMNS, build_features
from .model import Bundle, conformal_qhat, fit_lgb, predict_level, save_bundle, temp_groups
from .timeutils import mask_between
from .weather import apply_hourly_bias, hourly_bias


def train_bundle(
    df: pd.DataFrame, wt_fc: pd.DataFrame, train_start: str = "2013-01-01", train_end: pd.Timestamp | None = None,
    rounds: int = DEFAULT_ROUNDS, q_rounds: int = QUANTILE_ROUNDS, params: dict | None = None,
    calib_months: int = 12, bias_months: int = 24, alpha: float = 0.10, half_life: float | None = None,
    min_per_hour: int = 100, version: str | None = None,
) -> tuple[Bundle, dict]:
    """Entraîne le modèle de production. Renvoie le bundle et des diagnostics de calibration."""
    y = df["conso"]
    wt_obs = df[WCOLS]
    train_end = pd.Timestamp(train_end) if train_end is not None else consolidated_end(df)
    if train_end.tzinfo is None:
        train_end = train_end.tz_localize(TZ)
    cal_start = train_end - pd.DateOffset(months=calib_months)
    t0 = pd.Timestamp(train_start, tz=TZ)
    fc0 = pd.Timestamp(FC_START, tz=TZ)

    X_obs = build_features(df, wt_obs)

    # 1) modèles quantiles avant la fenêtre de calibration, scores sur la fenêtre (météo prévue débiaisée)
    bias_cal = hourly_bias(wt_fc, wt_obs, fc0, cal_start - pd.Timedelta(days=GAP_DAYS), min_per_hour)
    X_cal = build_features(df, apply_hourly_bias(wt_fc, bias_cal))
    tm_q = mask_between(X_obs.index, t0, cal_start - pd.Timedelta(days=GAP_DAYS))
    qp = {**(params or {}), "objective": "quantile", "metric": "quantile"}
    q_lo = fit_lgb(X_obs, y, tm_q, q_rounds, {**qp, "alpha": alpha / 2}, half_life, cal_start)
    q_hi = fit_lgb(X_obs, y, tm_q, q_rounds, {**qp, "alpha": 1 - alpha / 2}, half_life, cal_start)

    cal = mask_between(X_cal.index, cal_start, train_end) & X_cal["temp_nat"].notna().to_numpy() & y.notna().to_numpy()
    Xc = X_cal.loc[cal]
    lo = predict_level(q_lo, Xc)
    hi = predict_level(q_hi, Xc)
    lo, hi = lo.where(lo <= hi, hi), hi.where(hi >= lo, lo)
    scores = pd.concat([lo - y.loc[Xc.index], y.loc[Xc.index] - hi], axis=1).max(axis=1)
    groups = temp_groups(Xc["te"])
    qhat = conformal_qhat(scores, groups, alpha)
    inside_raw = ((y.loc[Xc.index] >= lo) & (y.loc[Xc.index] <= hi)).mean()

    # 2) biais horaire de production (mois récents) et modèle ponctuel sur toutes les données
    bias_prod = hourly_bias(wt_fc, wt_obs, train_end - pd.DateOffset(months=bias_months), train_end, min_per_hour)
    point = fit_lgb(X_obs, y, mask_between(X_obs.index, t0, train_end), rounds, params, half_life, train_end)

    meta = {
        "version": version or dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M"),
        "features": FEATURE_COLUMNS, "train_start": train_start, "train_end": str(train_end),
        "calibration_window": [str(cal_start), str(train_end)], "alpha": alpha, "rounds": rounds,
        "quantile_rounds": q_rounds, "half_life_years": half_life, "params": {**DEFAULT_PARAMS, **(params or {})},
        "target": "conso - level7 (moyenne des 7 jours complets T-8 ... T-2)", "issue_hour_local": 10,
        "weather": "entraîné sur météo observée ; servi avec météo prévue débiaisée par heure",
    }
    diag = {"coverage_before_calibration": float(inside_raw), "n_calibration_points": int(len(Xc)), "qhat": qhat}
    return Bundle(point, q_lo, q_hi, bias_prod, qhat, meta), diag


def main() -> None:  # pragma: no cover - point d'entrée en ligne de commande
    p = argparse.ArgumentParser(description="Entraîne le modèle de production et le sauvegarde.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--meteo-fc", required=True, help="prévisions météo historiques par ville (parquet)")
    p.add_argument("--out", default="models/prod")
    p.add_argument("--train-start", default="2013-01-01")
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    args = p.parse_args()

    df = load_dataset(args.dataset)
    wt_fc = load_national_forecast(args.meteo_fc, df.index)
    bundle, diag = train_bundle(df, wt_fc, train_start=args.train_start, rounds=args.rounds)
    save_bundle(bundle, args.out)
    print(f"✔ modèle enregistré dans {args.out} (version {bundle.meta['version']})")
    print(f"  couverture de l'intervalle brut sur la fenêtre de calibration : {diag['coverage_before_calibration']:.1%}")
    print(f"  élargissement conforme (MW) : { {k: round(v) for k, v in diag['qhat'].items()} }")


if __name__ == "__main__":  # pragma: no cover
    main()
