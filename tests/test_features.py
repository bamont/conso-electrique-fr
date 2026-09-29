import numpy as np
import pandas as pd

from conso.config import TZ, WCOLS
from conso.features import DYN_COLUMNS, FEATURE_COLUMNS, NOWCAST_COLUMNS, build_features, build_features_window
from conso.timeutils import day_bounds, local_days


def test_columns_and_level(dataset):
    X = build_features(dataset.loc["2023-01-01":"2023-03-01"], dataset.loc["2023-01-01":"2023-03-01", WCOLS])
    assert list(X.columns) == [*FEATURE_COLUMNS, "level7"]
    mid = X.drop(columns=["lag1_rel"]).iloc[-400:-100] # hors bords de la fenêtre (jour suivant inconnu)
    assert not mid.isna().any().any(), mid.columns[mid.isna().any()].tolist()


def test_no_leakage(dataset):
    """Émission le 11 mars à 10 h pour le 12 mars : effacer la consommation ensuite ne change rien."""
    T0 = pd.Timestamp("2023-03-12")
    cutoff = pd.Timestamp("2023-03-11 10:00", tz=TZ)
    sub = dataset.loc[pd.Timestamp("2023-01-15", tz=TZ):pd.Timestamp("2023-04-15", tz=TZ)]
    sub_mod = sub.copy()
    sub_mod.loc[sub_mod.index >= cutoff, "conso"] = np.nan
    Xa, Xb = build_features(sub, sub[WCOLS]), build_features(sub_mod, sub_mod[WCOLS])
    sel = local_days(sub.index) == T0
    assert sel.sum() == 48
    np.testing.assert_allclose(Xa.loc[sel].to_numpy(dtype=float), Xb.loc[sel].to_numpy(dtype=float), equal_nan=True)


def test_leakage_test_is_sensitive(dataset):
    """Garde-fou : le test ci-dessus détecte bien une fuite (effacer des données *antérieures* change les variables)."""
    T0 = pd.Timestamp("2023-03-12")
    sub = dataset.loc[pd.Timestamp("2023-01-15", tz=TZ):pd.Timestamp("2023-04-15", tz=TZ)]
    sub_mod = sub.copy()
    sub_mod.loc[sub_mod.index >= pd.Timestamp("2023-03-05", tz=TZ), "conso"] = np.nan # efface aussi T-7 ... T
    Xa, Xb = build_features(sub, sub[WCOLS]), build_features(sub_mod, sub_mod[WCOLS])
    sel = local_days(sub.index) == T0
    assert not np.allclose(Xa.loc[sel].to_numpy(dtype=float), Xb.loc[sel].to_numpy(dtype=float), equal_nan=True)


LONG_MEMOIRE = ["te_lent", "dT_te_lent"]
OPTIMISME_CONNU = {"te", "t_roll3", "dT"}


def test_window_equals_full_history(dataset):
    wt = dataset[WCOLS]
    X_full = build_features(dataset, wt)
    exactes = [c for c in X_full.columns if c not in LONG_MEMOIRE]
    for day in ["2023-03-15", "2023-10-29", "2024-03-31"]: # dont deux jours de changement d'heure
        start, end = day_bounds(day)
        X_win = build_features_window(dataset, wt, start, end)
        ref = X_full.loc[(X_full.index >= start) & (X_full.index < end)]
        assert len(X_win) == len(ref) in (46, 48, 50)
        np.testing.assert_allclose(X_win[exactes].to_numpy(dtype=float), ref[exactes].to_numpy(dtype=float), rtol=1e-6, atol=1e-6, equal_nan=True)
        np.testing.assert_allclose(X_win[LONG_MEMOIRE].to_numpy(dtype=float), ref[LONG_MEMOIRE].to_numpy(dtype=float), atol=0.03, equal_nan=True)


def _colonnes_modifiees(Xa, Xb, sel):
    return [c for c in Xa.columns if not np.allclose(Xa.loc[sel, c].to_numpy(dtype=float), Xb.loc[sel, c].to_numpy(dtype=float), equal_nan=True)]


def test_no_weather_leakage(dataset):
    """Effacer la météo observée à partir de D 10 h (la météo cible reste connue) ne change aucune nouvelle variable."""
    T0 = pd.Timestamp("2023-03-12")
    cutoff = pd.Timestamp("2023-03-11 10:00", tz=TZ)
    sub = dataset.loc[pd.Timestamp("2023-01-15", tz=TZ):pd.Timestamp("2023-04-15", tz=TZ)]
    wt = sub[WCOLS]
    sel = local_days(sub.index) == T0
    nouvelles = set(DYN_COLUMNS) | set(NOWCAST_COLUMNS)

    sub_mod = sub.copy()
    sub_mod.loc[sub_mod.index >= cutoff, WCOLS] = np.nan
    modifiees = set(_colonnes_modifiees(build_features(sub, wt), build_features(sub_mod, wt), sel))
    assert not modifiees & nouvelles, sorted(modifiees & nouvelles)
    assert modifiees - {"level7"} <= OPTIMISME_CONNU, sorted(modifiees)

    ancien = sub.copy()
    ancien.loc[ancien.index >= pd.Timestamp("2023-03-05", tz=TZ), WCOLS] = np.nan
    assert set(_colonnes_modifiees(build_features(sub, wt), build_features(ancien, wt), sel)) & nouvelles


def test_serie_froid_compte_les_jours_consecutifs(dataset):
    sub = dataset.loc[pd.Timestamp("2023-01-20", tz=TZ):pd.Timestamp("2023-03-01", tz=TZ)].copy()
    jours = local_days(sub.index)
    froid = (jours >= pd.Timestamp("2023-02-10")) & (jours <= pd.Timestamp("2023-02-13"))
    sub["temp_nat"] = np.where(froid, 2.0, 12.0)
    X = build_features(sub, sub[WCOLS])
    for jour, attendu in (("2023-02-10", 1), ("2023-02-11", 2), ("2023-02-13", 4), ("2023-02-14", 0)):
        assert (X.loc[jours == pd.Timestamp(jour), "serie_froid_5"] == attendu).all(), jour
    assert (X.loc[jours == pd.Timestamp("2023-02-12"), "serie_froid_0"] == 0).all()


def test_y_last_rel_est_le_dernier_point_connu(dataset):
    sub = dataset.loc[pd.Timestamp("2023-01-15", tz=TZ):pd.Timestamp("2023-04-15", tz=TZ)]
    X = build_features(sub, sub[WCOLS])
    T0 = pd.Timestamp("2023-03-12")
    sel = local_days(sub.index) == T0
    dernier = sub.loc[pd.Timestamp("2023-03-11 09:30", tz=TZ), "conso"]
    np.testing.assert_allclose(X.loc[sel, "y_last_rel"].to_numpy(), dernier - X.loc[sel, "level7"].to_numpy())
