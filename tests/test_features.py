import numpy as np
import pandas as pd

from conso.config import TZ, WCOLS
from conso.features import FEATURE_COLUMNS, build_features, build_features_window
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


def test_window_equals_full_history(dataset):
    wt = dataset[WCOLS]
    X_full = build_features(dataset, wt)
    for day in ["2023-03-15", "2023-10-29", "2024-03-31"]: # dont deux jours de changement d'heure
        start, end = day_bounds(day)
        X_win = build_features_window(dataset, wt, start, end)
        ref = X_full.loc[(X_full.index >= start) & (X_full.index < end)]
        assert len(X_win) == len(ref) in (46, 48, 50)
        np.testing.assert_allclose(X_win.to_numpy(dtype=float), ref.to_numpy(dtype=float), rtol=1e-6, atol=1e-6, equal_nan=True)
