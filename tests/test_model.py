import numpy as np
import pandas as pd

from conso.config import TEMP_GROUP_NAMES, WCOLS
from conso.features import build_features_window
from conso.model import conformal_qhat, load_bundle, predict_with_interval, save_bundle, temp_groups
from conso.timeutils import day_bounds
from conso.weather import apply_hourly_bias


def test_temp_groups_edges():
    g = temp_groups(pd.Series([-3.0, 4.9, 5.0, 14.9, 15.0, 21.9, 22.0, 30.0, np.nan]))
    assert list(g[:8]) == ["froid", "froid", "doux", "doux", "chaud", "chaud", "tres_chaud", "tres_chaud"]
    assert pd.isna(g.iloc[8])


def test_conformal_qhat_reaches_target_coverage():
    rng = np.random.default_rng(0)
    groups = pd.Series(rng.choice(list(TEMP_GROUP_NAMES), 20000))
    scores = pd.Series(rng.normal(0, 1, 20000) * groups.map({"froid": 3, "doux": 1, "chaud": 1, "tres_chaud": 2}).to_numpy())
    qhat = conformal_qhat(scores, groups, alpha=0.1)
    new_groups = pd.Series(rng.choice(list(TEMP_GROUP_NAMES), 20000))
    new_scores = pd.Series(rng.normal(0, 1, 20000) * new_groups.map({"froid": 3, "doux": 1, "chaud": 1, "tres_chaud": 2}).to_numpy())
    covered = new_scores <= new_groups.map(qhat)
    assert abs(covered.mean() - 0.9) < 0.01
    assert qhat["froid"] > qhat["doux"] # incertitude plus forte quand il fait froid


def test_bundle_structure(trained):
    bundle, diag = trained
    assert len(bundle.bias) == 24 and set(bundle.qhat) == set(TEMP_GROUP_NAMES)
    assert bundle.bias.between(0.0, 1.5).all() # biais synthétique ~ +0,3 à +1,1 °C
    assert 0.0 < diag["coverage_before_calibration"] < 1.0
    assert diag["n_calibration_points"] > 1000


def test_save_load_roundtrip(trained, dataset, wt_fc, tmp_path):
    bundle, _ = trained
    save_bundle(bundle, tmp_path / "m")
    loaded = load_bundle(tmp_path / "m")
    start, end = day_bounds("2024-03-20")
    X = build_features_window(dataset, apply_hourly_bias(wt_fc, bundle.bias), start, end)
    a, b = predict_with_interval(bundle, X), predict_with_interval(loaded, X)
    pd.testing.assert_frame_equal(a, b)
    assert loaded.meta["version"] == "test" and loaded.qhat == bundle.qhat


def test_interval_bounds_and_sanity(trained, dataset, wt_fc):
    bundle, _ = trained
    start, end = day_bounds("2024-03-20")
    X = build_features_window(dataset, apply_hourly_bias(wt_fc, bundle.bias), start, end)
    out = predict_with_interval(bundle, X)
    assert out.notna().all().all() and len(out) == 48
    assert (out["lower"] <= out["forecast"]).all() and (out["forecast"] <= out["upper"]).all()
    y = dataset.loc[out.index, "conso"]
    assert (out["forecast"] - y).abs().mean() < 3000 # ordre de grandeur cohérent
    assert set(WCOLS) <= set(dataset.columns)
