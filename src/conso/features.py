"""Construction des variables du modèle.

Toutes les variables d'un jour cible T n'utilisent que des informations disponibles à l'émission
(jour D = T-1, à 10 h) : le test ``tests/test_features.py::test_no_leakage`` le vérifie.

Deux jeux de météo interviennent :
- ``wt`` : météo au moment cible (observée en mode « oracle », prévue en exploitation) ;
- ``d[WCOLS]`` : météo passée (jours <= D), toujours observée.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HISTORY_DAYS, ISSUE_HOUR, TZ, WCOLS
from .timeutils import local_days, shift_local

FEATURE_COLUMNS = [
    "heure", "jour_semaine", "mois", "jour_annee", "is_holiday", "is_bridge",
    "vac_A", "vac_B", "vac_C", "n_zones_vac", "veille_ferie", "lendemain_ferie",
    "temp_nat", "hum_nat", "wind_nat", "cloud_nat", "rad_nat", "temp_l3h", "temp_l6h",
    "te", "t_roll3", "t_min_j", "t_max_j", "dT", "dtemp_7j", "rad_jour", "hdd7",
    "lag1_rel", "lag2_rel", "lag7_rel", "lag14_rel", "lag2_dev", "lag7_dev", "lag14_dev",
    "lag2_ferie", "lag2_pont", "lag7_ferie", "lag7_pont", "dev_matin",
]

REQUIRED_COLUMNS = ["conso", "is_holiday", "is_bridge", "vac_A", "vac_B", "vac_C", "n_zones_vac", *WCOLS]


def build_features(d: pd.DataFrame, wt: pd.DataFrame) -> pd.DataFrame:
    """Variables pour toutes les lignes de ``d`` (index UTC, pas de 30 min régulier).

    Renvoie les colonnes de ``FEATURE_COLUMNS`` plus ``level7`` (base de la cible : moyenne des
    7 jours complets T-8 ... T-2), qui n'est pas une variable du modèle.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in d.columns]
    if missing:
        raise KeyError(f"Colonnes manquantes dans d : {missing}")
    y_ = d["conso"]
    local = d.index.tz_convert(TZ)
    day = local_days(d.index)
    hh = np.asarray(local.hour + local.minute / 60)
    early = hh < ISSUE_HOUR

    def to_rows(s: pd.Series) -> pd.Series: 
        return pd.Series(s.reindex(day).to_numpy(), index=d.index)

    X = pd.DataFrame(index=d.index)

    X["heure"] = hh
    X["jour_semaine"] = local.dayofweek.to_numpy()
    X["mois"] = local.month.to_numpy()
    X["jour_annee"] = local.dayofyear.to_numpy()
    X["is_holiday"] = d["is_holiday"].astype(float)
    X["is_bridge"] = d["is_bridge"].astype(float)
    for c in ["vac_A", "vac_B", "vac_C", "n_zones_vac"]:
        X[c] = d[c]
    hol_d = d["is_holiday"].astype(float).groupby(day).max().asfreq("D")
    X["veille_ferie"] = to_rows(hol_d.shift(-1))
    X["lendemain_ferie"] = to_rows(hol_d.shift(1))

    for c in WCOLS:
        X[c] = wt[c]
    X["temp_l3h"] = wt["temp_nat"].shift(6)
    X["temp_l6h"] = wt["temp_nat"].shift(12)
    dt_obs = d["temp_nat"].groupby(day).mean().asfreq("D")
    dt_tgt = wt["temp_nat"].groupby(day).mean().asfreq("D")
    te_obs = dt_obs.ewm(alpha=0.5).mean().where(dt_obs.notna())
    te = to_rows(0.5 * dt_tgt + 0.5 * te_obs.shift(1))
    X["te"] = te
    X["t_roll3"] = to_rows((dt_tgt + dt_obs.shift(1) + dt_obs.shift(2)) / 3)
    X["t_min_j"] = to_rows(wt["temp_nat"].groupby(day).min().asfreq("D"))
    X["t_max_j"] = to_rows(wt["temp_nat"].groupby(day).max().asfreq("D"))
    X["dT"] = wt["temp_nat"] - te
    X["dtemp_7j"] = wt["temp_nat"] - shift_local(d["temp_nat"], 7)
    X["rad_jour"] = to_rows(wt["rad_nat"].groupby(day).mean().asfreq("D"))
    hdd_obs = (17 - te_obs).clip(lower=0)
    X["hdd7"] = to_rows(hdd_obs.shift(2).rolling(7, min_periods=7).mean())

    dy = y_.groupby(day).mean().asfreq("D")
    level7 = to_rows(dy.shift(2).rolling(7, min_periods=7).mean()) # jours T-8 ... T-2
    X["level7"] = level7
    lag = {k: shift_local(y_, k) for k in (1, 2, 7, 14)}
    lvl_prev = {k: shift_local(level7, k) for k in (2, 7, 14)}
    X["lag1_rel"] = (lag[1] - level7).where(early)
    X["lag2_rel"] = lag[2] - level7
    X["lag7_rel"] = lag[7] - level7
    X["lag14_rel"] = lag[14] - level7
    X["lag2_dev"] = lag[2] - lvl_prev[2]
    X["lag7_dev"] = lag[7] - lvl_prev[7]
    X["lag14_dev"] = lag[14] - lvl_prev[14]
    for k in (2, 7):
        X[f"lag{k}_ferie"] = shift_local(X["is_holiday"], k)
        X[f"lag{k}_pont"] = shift_local(X["is_bridge"], k)

    m_early = y_[early].groupby(day[early]).mean().asfreq("D")
    X["dev_matin"] = to_rows((m_early - m_early.shift(7)).shift(1))

    return X[[*FEATURE_COLUMNS, "level7"]]


def build_features_window(
    d: pd.DataFrame, wt: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, history_days: int = HISTORY_DAYS
) -> pd.DataFrame:
    """Variables des lignes de [start, end[, calculées sur une fenêtre (historique + jour suivant).

    Équivalent à ``build_features`` sur tout l'historique (vérifié par les tests), mais beaucoup plus
    rapide : c'est ce que fait l'API pour un jour donné, et le backtest pour chaque mois.
    """
    lo = start - pd.Timedelta(days=history_days)
    hi = end + pd.Timedelta(days=1) # le jour suivant sert à `veille_ferie`
    sel = (d.index >= lo) & (d.index < hi)
    X = build_features(d.loc[sel], wt.loc[sel])
    return X.loc[(X.index >= start) & (X.index < end)]
