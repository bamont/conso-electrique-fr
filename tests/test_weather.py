import numpy as np
import pandas as pd
import pytest

from conso.config import HOURLY, TZ, WCOLS
from conso.weather import apply_hourly_bias, hourly_bias, weighted_national


def _two_cities():
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC", name="time")
    a = pd.DataFrame({v: [10.0, 11.0, 12.0] for v in HOURLY}, index=idx).assign(city="Paris")
    b = pd.DataFrame({v: [20.0, np.nan, 22.0] for v in HOURLY}, index=idx).assign(city="Lyon")
    return pd.concat([a, b])


def test_weighted_national_ignores_missing_city():
    nat = weighted_national(_two_cities(), weights={"Paris": 3.0, "Lyon": 1.0})
    assert list(nat.columns) == WCOLS
    assert np.isclose(nat["temp_nat"].iloc[0], (3 * 10 + 20) / 4)
    assert np.isclose(nat["temp_nat"].iloc[1], 11.0) # Lyon manquante : seule Paris compte


def test_hourly_bias_recovers_and_removes_bias():
    idx = pd.date_range("2023-01-01", "2023-12-31 23:00", freq="h", tz="UTC")
    hours = idx.tz_convert(TZ).hour
    obs = pd.DataFrame({c: 0.0 for c in WCOLS}, index=idx)
    true_bias = 0.5 + 0.5 * np.cos(2 * np.pi * hours / 24)
    fc = obs.copy()
    fc["temp_nat"] = obs["temp_nat"] + true_bias
    b = hourly_bias(fc, obs, idx[0], idx[-1] + pd.Timedelta(hours=1))
    assert len(b) == 24 and np.allclose(b.to_numpy(), 0.5 + 0.5 * np.cos(2 * np.pi * np.arange(24) / 24), atol=1e-9)
    fixed = apply_hourly_bias(fc, b)
    assert np.allclose(fixed["temp_nat"], 0.0, atol=1e-9)
    assert (fixed["hum_nat"] == fc["hum_nat"]).all() # seule la température est corrigée


def test_hourly_bias_needs_enough_points():
    idx = pd.date_range("2023-01-01", periods=48, freq="h", tz="UTC")
    obs = pd.DataFrame({c: 0.0 for c in WCOLS}, index=idx)
    with pytest.raises(ValueError):
        hourly_bias(obs, obs, idx[0], idx[-1])
