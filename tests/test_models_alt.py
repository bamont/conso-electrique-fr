import numpy as np
import pandas as pd
import pytest

from conso.config import WCOLS
from conso.features import FEATURE_COLUMNS, build_features
from conso.model import Bundle, load_bundle, predict_with_interval, save_bundle
from conso.models_alt import (
    HAS_CATBOOST,
    HAS_TORCH,
    Blend,
    CatBoostSeed,
    Hybride,
    LgbmSeedPoint,
    Lisse,
    MLPPoint,
    fit_blend,
)
from conso.timeutils import day_bounds, mask_between

PETITS_PARAMS = {"num_leaves": 15, "min_data_in_leaf": 50, "learning_rate": 0.2}


@pytest.fixture(scope="module")
def petit_X(dataset):
    sub = dataset.loc["2023-01-01":"2023-06-30"]
    X = build_features(sub, sub[WCOLS])
    masque = mask_between(X.index, X.index[0], X.index[-1] - pd.Timedelta(days=10))
    return sub, X, masque


def test_lisse_fit_predict_save_load(petit_X, tmp_path):
    sub, X, masque = petit_X
    m = Lisse().fit(X, sub["conso"], masque)
    p1 = m.predict(X[FEATURE_COLUMNS])
    assert np.isfinite(p1).mean() > 0.95
    m.save(tmp_path / "lisse")
    m2 = Lisse.load(tmp_path / "lisse")
    np.testing.assert_allclose(m2.predict(X[FEATURE_COLUMNS]), p1)


def test_lisse_extrapole_au_dela_du_froid_vu(petit_X):
    sub, X, masque = petit_X
    seuil = sub["temp_nat"].quantile(0.2)
    pas_froid = masque & (sub["temp_nat"].to_numpy() >= seuil)
    m = Lisse().fit(X, sub["conso"], pas_froid)
    froid = X.loc[(sub["temp_nat"] < seuil - 5).to_numpy()]
    assert len(froid) > 0
    p = m.predict(froid[FEATURE_COLUMNS])
    assert np.isfinite(p).mean() > 0.9


def test_hybride_fit_predict_save_load(petit_X, tmp_path):
    sub, X, masque = petit_X
    m = Hybride(rounds=30, seeds=1).fit(X, sub["conso"], masque)
    p1 = m.predict(X[FEATURE_COLUMNS])
    m.save(tmp_path / "hybride")
    m2 = Hybride.load(tmp_path / "hybride")
    np.testing.assert_allclose(m2.predict(X[FEATURE_COLUMNS]), p1)


def test_lgbm_seed_point_save_load(petit_X, tmp_path):
    sub, X, masque = petit_X
    m = LgbmSeedPoint().fit(X, sub["conso"], masque, rounds=30, seeds=2, params=PETITS_PARAMS)
    p1 = m.predict(X[FEATURE_COLUMNS])
    m.save(tmp_path / "lgbm")
    m2 = LgbmSeedPoint.load(tmp_path / "lgbm")
    np.testing.assert_allclose(m2.predict(X[FEATURE_COLUMNS]), p1)


@pytest.mark.skipif(not HAS_CATBOOST, reason="catboost non installé")
def test_catboost_seed_fit_predict_save_load(petit_X, tmp_path):
    sub, X, masque = petit_X
    m = CatBoostSeed(iters=20, depth=4, seeds=2).fit(X, sub["conso"], masque)
    p1 = m.predict(X[FEATURE_COLUMNS])
    assert np.isfinite(p1).all()
    m.save(tmp_path / "cat")
    m2 = CatBoostSeed.load(tmp_path / "cat")
    np.testing.assert_allclose(m2.predict(X[FEATURE_COLUMNS]), p1, rtol=1e-5)


@pytest.mark.skipif(not HAS_TORCH, reason="torch non installé")
def test_mlp_fit_predict_save_load(petit_X, tmp_path):
    sub, X, masque = petit_X
    m = MLPPoint(seeds=1, epochs=2, hidden=16).fit(X, sub["conso"], masque)
    p1 = m.predict(X[FEATURE_COLUMNS])
    assert np.isfinite(p1).all()
    m.save(tmp_path / "mlp")
    m2 = MLPPoint.load(tmp_path / "mlp")
    np.testing.assert_allclose(m2.predict(X[FEATURE_COLUMNS]), p1, rtol=1e-5)


def test_blend_est_la_somme_ponderee(petit_X, tmp_path):
    sub, X, masque = petit_X
    lisse = Lisse().fit(X, sub["conso"], masque)
    lgbm = LgbmSeedPoint().fit(X, sub["conso"], masque, rounds=30, seeds=1, params=PETITS_PARAMS)
    poids = {"lisse": 0.4, "lgbm": 0.6}
    b = Blend({"lisse": lisse, "lgbm": lgbm}, poids)
    Xf = X[FEATURE_COLUMNS]
    attendu = 0.4 * lisse.predict(Xf) + 0.6 * lgbm.predict(Xf)
    np.testing.assert_allclose(b.predict(Xf), attendu)

    b.save(tmp_path / "blend")
    b2 = Blend.load(tmp_path / "blend")
    np.testing.assert_allclose(b2.predict(Xf), attendu)
    assert b2.weights == poids


def test_blend_ignore_les_membres_a_poids_nul(petit_X):
    sub, X, masque = petit_X
    b = fit_blend(X, sub["conso"], masque, weights={"lgbm": 1.0, "lisse": 0.0}, seeds=1, rounds=20)
    assert set(b.models) == {"lgbm"} # lisse à poids nul : mesuré nulle part, pas entraîné


def test_fit_blend_rejette_membre_inconnu(petit_X):
    sub, X, masque = petit_X
    with pytest.raises(ValueError):
        fit_blend(X, sub["conso"], masque, weights={"modele_qui_n_existe_pas": 1.0})


def test_bundle_avec_blend_round_trip(petit_X, dataset, wt_fc, tmp_path):
    sub, X, masque = petit_X
    ref_end = X.index[-1] - pd.Timedelta(days=10)
    point = fit_blend(X, sub["conso"], masque, ref_end, weights={"lgbm": 0.6, "lisse": 0.4}, seeds=1, rounds=20)
    q_lo = LgbmSeedPoint().fit(X, sub["conso"], masque, ref_end, rounds=20, seeds=1, params={**PETITS_PARAMS, "objective": "quantile", "alpha": 0.05})
    q_hi = LgbmSeedPoint().fit(X, sub["conso"], masque, ref_end, rounds=20, seeds=1, params={**PETITS_PARAMS, "objective": "quantile", "alpha": 0.95})
    bias = pd.Series(0.5, index=range(24))
    bundle = Bundle(point, q_lo.model, q_hi.model, bias, {"froid": 500.0, "doux": 400.0, "chaud": 400.0, "tres_chaud": 450.0}, {"point_model": "blend"})

    save_bundle(bundle, tmp_path / "m")
    assert (tmp_path / "m" / "point" / "blend.json").exists()
    loaded = load_bundle(tmp_path / "m")
    assert isinstance(loaded.point, Blend)
    assert loaded.meta["point_kind"] == "blend"

    start, end = day_bounds("2023-04-10")
    Xw = build_features(dataset.loc[:end + pd.Timedelta(days=2)], dataset.loc[:end + pd.Timedelta(days=2), WCOLS])
    Xw = Xw.loc[(Xw.index >= start) & (Xw.index < end)]
    a, b = predict_with_interval(bundle, Xw), predict_with_interval(loaded, Xw)
    pd.testing.assert_frame_equal(a, b)


def test_train_bundle_point_model_blend(dataset, wt_fc, tmp_path):
    """Le chemin complet de production (``train_bundle``) avec ``point_model=\"blend\"``."""
    from conso.train import train_bundle

    bundle, diag = train_bundle(
        dataset, wt_fc, train_start="2021-01-01", train_end=pd.Timestamp("2024-04-01", tz="Europe/Paris"),
        rounds=20, q_rounds=20, params=PETITS_PARAMS, calib_months=6, bias_months=12, min_per_hour=50,
        version="test-blend", seeds=1, point_model="blend", blend_weights={"lgbm": 0.6, "lisse": 0.4},
    )
    assert isinstance(bundle.point, Blend) and bundle.meta["point_model"] == "blend"
    save_bundle(bundle, tmp_path / "m")
    loaded = load_bundle(tmp_path / "m")
    start, end = day_bounds("2024-03-20")
    Xw = build_features(dataset, dataset[WCOLS])
    Xw = Xw.loc[(Xw.index >= start) & (Xw.index < end)]
    pd.testing.assert_frame_equal(predict_with_interval(bundle, Xw), predict_with_interval(loaded, Xw))


def test_rolling_backtest_point_model_blend(dataset, wt_fc):
    from conso.backtest import rolling_backtest

    bt = rolling_backtest(
        dataset, wt_fc, pd.Timestamp("2024-01-01", tz="Europe/Paris"), pd.Timestamp("2024-03-01", tz="Europe/Paris"),
        train_start="2021-01-01", rounds=20, params=PETITS_PARAMS, seeds=1, progress=False,
        point_model="blend", blend_weights={"lgbm": 0.6, "lisse": 0.4},
    )
    assert set(bt.columns) == {"oracle", "fc", "fc_debiased"}
    assert bt["oracle"].notna().mean() > 0.9


def test_load_bundle_ancien_format_reste_lgbm(trained, tmp_path):
    bundle, _ = trained
    save_bundle(bundle, tmp_path / "m")
    meta_path = tmp_path / "m" / "meta.json"
    import json

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["point_kind"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    loaded = load_bundle(tmp_path / "m")
    assert not isinstance(loaded.point, Blend)
