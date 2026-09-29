"""Construction des variables du modèle.

Toutes les variables d'un jour cible T n'utilisent que des informations disponibles à l'émission
(jour D = T-1, à 10 h).

Deux jeux de météo interviennent :
- ``wt`` : météo au moment cible (observée en mode « oracle », prévue en exploitation) ;
- ``d[WCOLS]`` : météo passée (jours <= D), toujours observée.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HISTORY_DAYS, ISSUE_HOUR, TZ, WCOLS
from .timeutils import local_days, shift_local

BASE_COLUMNS = [
    "heure", "jour_semaine", "mois", "jour_annee", "is_holiday", "is_bridge",
    "vac_A", "vac_B", "vac_C", "n_zones_vac", "veille_ferie", "lendemain_ferie",
    "temp_nat", "hum_nat", "wind_nat", "cloud_nat", "rad_nat", "temp_l3h", "temp_l6h",
    "te", "t_roll3", "t_min_j", "t_max_j", "dT", "dtemp_7j", "rad_jour", "hdd7",
    "lag1_rel", "lag2_rel", "lag7_rel", "lag14_rel", "lag2_dev", "lag7_dev", "lag14_dev",
    "lag2_ferie", "lag2_pont", "lag7_ferie", "lag7_pont", "dev_matin",
]

DYN_COLUMNS = [
    "dtemp_24h", "dtemp_48h", "dTj_1j", "dTj_3j", "te_lent", "te_moyen", "te_rapide", "dT_te_lent",
    "tmin_3j", "serie_froid_0", "serie_froid_5",
]
NOWCAST_COLUMNS = ["y_last_rel", "pente_matin_dev", "dev_recent_3h", "dev_matin_j1", "dtemp_matin"]
FEATURE_COLUMNS = [*BASE_COLUMNS, *DYN_COLUMNS, *NOWCAST_COLUMNS]

REQUIRED_COLUMNS = ["conso", "is_holiday", "is_bridge", "vac_A", "vac_B", "vac_C", "n_zones_vac", *WCOLS]


def _dynamique(d, wt, day, to_rows, dm, do) -> pd.DataFrame:
    """Dynamique thermique : moyennes exponentielles multi-échelles, variations et séries de jours froids.

    ``dm`` : température moyenne journalière au moment cible (prévue en exploitation) ;
    ``do`` : température moyenne journalière observée. Le jour T et le jour D viennent de ``dm``,
    les jours antérieurs de ``do``.
    """
    t = wt["temp_nat"]
    o = pd.DataFrame(index=d.index)
    o["dtemp_24h"] = t - shift_local(t, 1)
    o["dtemp_48h"] = t - shift_local(d["temp_nat"], 2)
    p1, p2, p3 = dm.shift(1), do.shift(2), do.shift(3)
    o["dTj_1j"] = to_rows(dm - p1)
    o["dTj_3j"] = to_rows(dm - (p1 + p2 + p3) / 3)
    for alpha, nom in ((0.15, "lent"), (0.35, "moyen"), (0.7, "rapide")):
        ew = do.ewm(alpha=alpha, adjust=False).mean().where(do.notna())
        etat_d = alpha * dm.shift(1) + (1 - alpha) * ew.shift(2)
        o[f"te_{nom}"] = to_rows(alpha * dm + (1 - alpha) * etat_d)
    o["dT_te_lent"] = t - o["te_lent"]
    mn_cible = t.groupby(day).min().asfreq("D")
    mn_obs = d["temp_nat"].groupby(day).min().asfreq("D")
    o["tmin_3j"] = to_rows(pd.concat([mn_cible, mn_cible.shift(1), mn_obs.shift(2)], axis=1).min(axis=1, skipna=False))
    for seuil in (0, 5):
        froid = (do < seuil).astype(int)
        serie_obs = froid.groupby((1 - froid).cumsum()).cumsum().where(do.notna())
        s_t = (dm < seuil).astype(float).where(dm.notna())
        s_d = (dm.shift(1) < seuil).astype(float).where(dm.shift(1).notna())
        o[f"serie_froid_{seuil}"] = to_rows(s_t * (1 + s_d * (1 + serie_obs.shift(2).fillna(0))))
    return o


def _nowcast(d, day, hh, early, to_rows, level7) -> pd.DataFrame:
    """Informations de la matinée du jour D (avant l'heure d'émission) sur le comportement du jour T."""
    y_ = d["conso"]

    def creneau(h: float) -> pd.Series:
        k = np.isclose(hh, h)
        return y_[k].groupby(day[k]).first().asfreq("D")

    y_fin, y_debut = creneau(ISSUE_HOUR - 0.5), creneau(ISSUE_HOUR - 2.5)
    o = pd.DataFrame(index=d.index)
    o["y_last_rel"] = to_rows(y_fin.shift(1)) - level7
    pente = y_fin - y_debut
    o["pente_matin_dev"] = to_rows((pente - pente.shift(7)).shift(1))
    rec = (hh >= ISSUE_HOUR - 3) & early
    m_rec = y_[rec].groupby(day[rec]).mean().asfreq("D")
    o["dev_recent_3h"] = to_rows((m_rec - m_rec.shift(7)).shift(1))
    m_early = y_[early].groupby(day[early]).mean().asfreq("D")
    o["dev_matin_j1"] = to_rows((m_early - m_early.shift(1)).shift(1))
    t_early = d["temp_nat"][early].groupby(day[early]).mean().asfreq("D")
    o["dtemp_matin"] = to_rows((t_early - t_early.shift(7)).shift(1))
    return o


def build_features(d: pd.DataFrame, wt: pd.DataFrame) -> pd.DataFrame:
    """Variables pour toutes les lignes de ``d`` (index UTC, pas de 30 min régulier).

    Renvoie les colonnes de ``FEATURE_COLUMNS`` plus ``level7`` (base de la cible : moyenne des
    7 jours complets T-8 à T-2).
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

    X = pd.concat([X, _dynamique(d, wt, day, to_rows, dt_tgt, dt_obs), _nowcast(d, day, hh, early, to_rows, level7)], axis=1)
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
