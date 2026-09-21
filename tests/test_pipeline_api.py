import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from conso.api.main import create_app
from conso.pipeline import apply_issue_cutoff, forecast_day, issue_time
from conso.providers import Inputs, LiveProvider, ReplayProvider, parse_rte_export


@pytest.fixture(scope="module")
def replay(dataset, wt_fc):
    return ReplayProvider(dataset, wt_fc)


def test_issue_time_is_previous_day_10h():
    assert issue_time("2024-03-20") == pd.Timestamp("2024-03-19 10:00", tz="Europe/Paris")


def test_forecast_day_is_causal(trained, replay):
    """Ce qui est connu après l'émission ne doit pas influencer la prévision, même si on le fournit."""
    bundle, _ = trained
    inp = replay.get_inputs("2024-03-20")
    a = forecast_day(bundle, inp.d, inp.wt_raw, "2024-03-20")
    d2 = inp.d.copy()
    later = d2.index >= issue_time("2024-03-20")
    d2.loc[later, "conso"] = np.random.default_rng(0).uniform(0, 1e5, later.sum()) # « futur » aberrant
    b = forecast_day(bundle, d2, inp.wt_raw, "2024-03-20")
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 48 and a.notna().all().all()
    assert apply_issue_cutoff(d2, "2024-03-20").loc[later, "conso"].isna().all()


def test_forecast_needs_history(trained, replay):
    bundle, _ = trained
    inp = replay.get_inputs("2024-03-20")
    with pytest.raises(ValueError):
        forecast_day(bundle, inp.d.loc["2024-03-15":], inp.wt_raw.loc["2024-03-15":], "2024-03-20")


class FakeLive:
    def __init__(self, replay, fail=False):
        self.replay, self.fail = replay, fail

    def get_inputs(self, target_date, bias=None):
        if self.fail:
            raise RuntimeError("réseau indisponible")
        return self.replay.get_inputs(target_date, bias)


@pytest.fixture(scope="module")
def client(trained, replay):
    return TestClient(create_app(trained[0], replay, FakeLive(replay)))


def test_health_and_model(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True, "replay_available": True}
    m = client.get("/model").json()
    assert m["version"] == "test" and len(m["bias_hourly_c"]) == 24 and "features" not in m


def test_forecast_replay(client):
    r = client.get("/forecast", params={"date": "2024-03-20"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["n_points"] == 48 and j["mode"] == "replay" and j["issue_time"].startswith("2024-03-19T10:00")
    p = j["points"]
    assert p[0]["time"].startswith("2024-03-20T00:00") and p[-1]["time"].startswith("2024-03-20T23:30")
    assert all(x["lower_mw"] <= x["forecast_mw"] <= x["upper_mw"] for x in p)
    assert all(x["actual_mw"] is not None and x["rte_j1_mw"] is not None for x in p)


def test_forecast_dst_day_has_46_points(client):
    r = client.get("/forecast", params={"date": "2024-03-31"})
    assert r.status_code == 200 and r.json()["n_points"] == 46


def test_forecast_errors(client):
    assert client.get("/forecast", params={"date": "2019-01-01"}).status_code == 404       # hors couverture
    assert client.get("/forecast", params={"date": "2021-01-10"}).status_code == 404       # moins de 45 jours d'historique
    assert client.get("/forecast", params={"date": "2021-12-15"}).status_code == 404       # avant les prévisions météo (2022)
    assert client.get("/forecast", params={"date": "20-03-2024"}).status_code == 422       # format invalide
    assert client.get("/forecast", params={"date": "2024-03-20", "mode": "autre"}).status_code == 422


def test_live_mode_and_failure(trained, replay):
    ok = TestClient(create_app(trained[0], replay, FakeLive(replay)))
    assert ok.get("/forecast", params={"date": "2024-03-20", "mode": "live"}).status_code == 200
    ko = TestClient(create_app(trained[0], replay, FakeLive(replay, fail=True)))
    assert ko.get("/forecast", params={"date": "2024-03-20", "mode": "live"}).status_code == 503


def test_without_model_or_replay(trained, replay):
    empty = TestClient(create_app(None, None))
    assert empty.get("/health").json()["model_loaded"] is False
    assert empty.get("/forecast", params={"date": "2024-03-20"}).status_code == 503
    no_replay = TestClient(create_app(trained[0], None))
    assert no_replay.get("/forecast", params={"date": "2024-03-20"}).status_code == 503


# Données simulées pour tester le parsing de l'export RTE (CSV) et la récupération de la météo (JSON horaire).
RTE_CSV = "perimetre;nature;date_heure;consommation\nFrance;Données temps réel;{ts};{v}\n"


def test_parse_rte_export_keeps_half_hours():
    lines = ["perimetre;nature;date_heure;consommation"]
    for i, t in enumerate(pd.date_range("2024-03-01", periods=8, freq="15min", tz="UTC")):
        lines.append(f"France;Données temps réel;{t.strftime('%Y-%m-%dT%H:%M:%S+00:00')};{50000 + i}")
    lines.append("France;Données temps réel;2024-03-01T02:00:00+00:00;") # valeur manquante
    s = parse_rte_export("\n".join(lines))
    assert list(s.index.minute) == [0, 30, 0, 30] and s.iloc[1] == 50002


class FakeResponse:
    def __init__(self, status=200, text="", payload=None):
        self.status_code, self.text, self._payload = status, text, payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Simule les sources externes : export RTE (CSV) et Open-Meteo (JSON horaire)."""

    def __init__(self, now):
        self.now = now

    def get(self, url, params=None, timeout=None):
        if "opendata.reseaux-energies" in url:
            idx = pd.date_range(self.now - pd.Timedelta(days=70), self.now, freq="15min")
            rows = ["perimetre;nature;date_heure;consommation"]
            rows += [f"France;Données temps réel;{t.strftime('%Y-%m-%dT%H:%M:%S+00:00')};{50000 + 5000 * np.sin(t.hour / 24 * 6.28):.0f}" for t in idx]
            return FakeResponse(text="\n".join(rows))
        past, fut = params["past_days"], params["forecast_days"]
        idx = pd.date_range((self.now - pd.Timedelta(days=past)).floor("D"), (self.now + pd.Timedelta(days=fut)).ceil("D"), freq="h")
        h = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in idx]}
        for v, val in zip(params["hourly"].split(","), [10.0, 70.0, 10.0, 50.0, 100.0], strict=True):
            h[v] = [val + np.sin(i / 24 * 6.28) for i in range(len(idx))]
        return FakeResponse(payload={"hourly": h})


def test_live_provider_assembles_inputs(trained):
    now = pd.Timestamp.now(tz="UTC").floor("15min")
    target = (pd.Timestamp.now(tz="Europe/Paris").normalize().tz_localize(None) + pd.Timedelta(days=1)).date()
    provider = LiveProvider(session=FakeSession(now))
    inp = provider.get_inputs(target, trained[0].bias)
    assert isinstance(inp, Inputs) and inp.actual is None
    fc = forecast_day(trained[0], inp.d, inp.wt_raw, target)
    assert len(fc) in (46, 48, 50) and fc.notna().all().all()


def test_live_provider_rejects_incomplete_history(trained):
    class Short(FakeSession):
        def get(self, url, params=None, timeout=None):
            if "opendata.reseaux-energies" in url:
                idx = pd.date_range(self.now - pd.Timedelta(days=3), self.now, freq="15min")
                return FakeResponse(text="perimetre;nature;date_heure;consommation\n" + "\n".join(
                    f"France;x;{t.strftime('%Y-%m-%dT%H:%M:%S+00:00')};50000" for t in idx))
            return super().get(url, params, timeout)

    now = pd.Timestamp.now(tz="UTC").floor("15min")
    target = (pd.Timestamp.now(tz="Europe/Paris").normalize().tz_localize(None) + pd.Timedelta(days=1)).date()
    with pytest.raises(LookupError):
        LiveProvider(session=Short(now)).get_inputs(target, trained[0].bias)
