"""Tests du tableau de bord : fonctions pures, client HTTP, et pages Streamlit branchées sur la vraie API (TestClient)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from fastapi.testclient import TestClient  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from conso.api.main import create_app  # noqa: E402
from conso.dashboard import charts, client, performance  # noqa: E402
from conso.providers import ReplayProvider  # noqa: E402

APP = str(Path(__file__).parents[1] / "src" / "conso" / "dashboard" / "app.py")


def _points(n=48, with_reference=True):
    idx = pd.date_range("2025-01-15", periods=n, freq="30min", tz="Europe/Paris")
    base = 60000 + 5000 * np.sin(np.arange(n) / 8)
    df = pd.DataFrame({"forecast_mw": base, "lower_mw": base - 2000, "upper_mw": base + 2000}, index=idx)
    if with_reference:
        df["actual_mw"], df["rte_j1_mw"] = base + 300, base - 800
    return df


def test_forecast_figure_traces():
    fig = charts.forecast_figure(_points(), "titre")
    assert [t.name for t in fig.data] == [None, "Intervalle à 90 %", "Prévision", "RTE J-1", "Réel"]
    assert fig.data[2].y[0] == pytest.approx(_points()["forecast_mw"].iloc[0] / 1000)
    live = charts.forecast_figure(_points(with_reference=False))
    assert [t.name for t in live.data] == [None, "Intervalle à 90 %", "Prévision"] # pas de réel ni de RTE en direct


def test_bar_line_and_bias_figures():
    t = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 1.0]}, index=["x", "y"])
    assert len(charts.lines_figure(t, "t", "u").data) == 2
    assert len(charts.bars_figure(t, "t", "u").data) == 2
    assert len(charts.hourly_bias_figure({h: 0.5 for h in range(24)}).data[0].x) == 24


@pytest.fixture(scope="module")
def perf_data(dataset):
    rng = np.random.default_rng(0)
    sub = dataset.loc["2023-06-01":"2024-04-30"]
    y = sub["conso"]
    bt = pd.DataFrame({"oracle": y + rng.normal(0, 800, len(y)), "fc": y + 500 + rng.normal(0, 1200, len(y)),
                       "fc_debiased": y + rng.normal(0, 900, len(y))}, index=sub.index)
    return dataset, bt.loc["2024-01-01":]


def test_performance_tables(perf_data):
    df, bt = perf_data
    F = performance.eval_frame(df, bt)
    assert list(F.columns) == ["y", "temp_nat", *performance.LABELS]
    s = performance.summary(F, scale=1000.0)
    manual = (F["prod_fc_debiased"] - F["y"]).abs().mean()
    assert s.loc[performance.LABELS["prod_fc_debiased"], "MAE (MW)"] == pytest.approx(manual)
    assert s.loc[performance.LABELS["prod_fc_debiased"], "MASE"] == pytest.approx(manual / 1000.0)
    assert s.loc[performance.LABELS["prod_fc"], "Biais (MW)"] > 300 # le biais synthétique de +500 MW se voit
    assert set(performance.mae_by_slot(F).index) == {"0 h - 9 h 30", "10 h - 23 h 30"}
    assert list(performance.mae_by_month(F).index)[:4] == ["2024-01", "2024-02", "2024-03", "2024-04"] # mois en heure locale
    by_t = performance.mae_by_temperature(F)
    assert set(by_t.index) <= set(performance.TEMP_LABELS) and by_t.notna().all().all()
    ry = performance.rte_by_year(df)
    assert list(ry.columns) == ["MAPE RTE J-1 (%)", "Biais RTE J-1 (MW)"] and 2021 in ry.index


# Client HTTP
class Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code, self._p, self.text = status, payload, text

    def json(self):
        if self._p is None:
            raise ValueError("pas de JSON")
        return self._p


def test_client_error_mapping(monkeypatch):
    monkeypatch.setattr(client.requests, "get", lambda *a, **k: Resp(404, {"detail": "introuvable"}))
    with pytest.raises(client.ApiError) as e:
        client.get_json("/x")
    assert e.value.status == 404 and "introuvable" in str(e.value)

    monkeypatch.setattr(client.requests, "get", lambda *a, **k: Resp(500, None, "boum"))
    with pytest.raises(client.ApiError, match="boum"):
        client.get_json("/x")

    def boom(*a, **k):
        raise requests.ConnectionError("refusé")

    monkeypatch.setattr(client.requests, "get", boom)
    with pytest.raises(client.ApiError, match="injoignable") as e:
        client.get_json("/x", base_url="http://api:8000")
    assert e.value.status is None


def test_forecast_frame_is_local_time():
    payload = {"points": [{"time": "2024-03-31T01:30:00+01:00", "forecast_mw": 1.0, "lower_mw": 0.0, "upper_mw": 2.0},
                          {"time": "2024-03-31T03:00:00+02:00", "forecast_mw": 1.0, "lower_mw": 0.0, "upper_mw": 2.0}]}
    df = client.forecast_frame(payload)
    assert str(df.index.tz) == "Europe/Paris" and df.index[1] - df.index[0] == pd.Timedelta("30min") # saut d'heure d'été


# Pages Streamlit branchées sur la vraie API
class FixedLive:
    """« Live » simulé : rejoue l'historique pour le jour demandé."""

    def __init__(self, replay):
        self.replay = replay

    def get_inputs(self, target_date, bias=None):
        return self.replay.get_inputs(target_date, bias)


@pytest.fixture()
def wired(monkeypatch, trained, dataset, wt_fc):
    """Redirige les appels HTTP du dashboard vers l'API FastAPI (TestClient), et fige l'heure au 19/03/2024 14 h."""
    import streamlit as st

    st.cache_data.clear()
    replay = ReplayProvider(dataset, wt_fc)
    api = TestClient(create_app(trained[0], replay, FixedLive(replay)))

    def fake_get(url, params=None, timeout=None):
        return api.get(url.replace(client.api_url(), ""), params=params)

    monkeypatch.setattr(client.requests, "get", fake_get)
    monkeypatch.setattr(client, "now_paris", lambda: pd.Timestamp("2024-03-19 14:00", tz="Europe/Paris"))
    return monkeypatch


def _run():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def _labels(at):
    return [m.label for m in at.metric]


def test_live_page(wired):
    at = _run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Pointe prévue" in _labels(at) and len(at.metric) == 4
    assert at.get("plotly_chart") and not at.warning and not at.error
    assert "mercredi 20/03/2024" in " ".join(c.value for c in at.caption)
    assert any("connectée" in s.value for s in at.sidebar.success)


def test_live_page_warns_before_issue_hour(wired):
    wired.setattr(client, "now_paris", lambda: pd.Timestamp("2024-03-19 08:00", tz="Europe/Paris"))
    at = _run()
    assert not at.exception and any("avant 10 h" in w.value for w in at.warning)


def test_replay_page(wired):
    at = _run()
    at.sidebar.radio[0].set_value("Rejouer un jour").run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Erreur moyenne du modèle" in _labels(at) and len(at.metric) == 4
    assert at.date_input[0].value.isoformat() == "2024-04-30" # dernier jour rejouable (borne de l'API)


def test_model_page(wired):
    at = _run()
    at.sidebar.radio[0].set_value("Modèle").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.metric[0].value == "test" and len(at.get("plotly_chart")) == 2


def test_performance_page(wired, perf_data, tmp_path):
    df, bt = perf_data
    ds, btp = tmp_path / "d.parquet", tmp_path / "bt.parquet"
    df[["conso", "prevision_j1", "temp_nat"]].to_parquet(ds)
    bt.to_parquet(btp)
    wired.setenv("CONSO_DATASET", str(ds))
    wired.setenv("CONSO_BACKTEST", str(btp))
    at = _run()
    at.sidebar.radio[0].set_value("Performance").run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.metric) == 3 and len(at.dataframe) == 1 and len(at.get("plotly_chart")) == 4


def test_performance_page_without_files(wired, tmp_path):
    wired.setenv("CONSO_DATASET", str(tmp_path / "absent.parquet"))
    at = _run()
    at.sidebar.radio[0].set_value("Performance").run()
    assert not at.exception and any("introuvables" in i.value for i in at.info)


def test_api_down(monkeypatch):
    import streamlit as st

    st.cache_data.clear()

    def boom(*a, **k):
        raise requests.ConnectionError("refusé")

    monkeypatch.setattr(client.requests, "get", boom)
    monkeypatch.setattr(client, "now_paris", lambda: pd.Timestamp("2024-03-19 14:00", tz="Europe/Paris"))
    at = _run()
    assert not at.exception # dégradation propre, pas d'exception
    assert at.sidebar.error and any("indisponible" in e.value for e in at.error)
