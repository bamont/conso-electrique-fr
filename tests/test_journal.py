"""Tests du journal des prévisions réelles, avec une session RTE / Open-Meteo simulée (hors ligne)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conso import journal
from conso.config import TZ
from conso.providers import LiveProvider


class FakeResponse:
    def __init__(self, status=200, text="", payload=None):
        self.status_code, self.text, self._payload = status, text, payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Simule l'export RTE (consommation + prévisions J-1/J) et les prévisions météo Open-Meteo."""

    def __init__(self, now: pd.Timestamp, bias_rte: float = 300.0, span_days: int = 75):
        # aligné sur 15 min : un pas de 15 min conserve alors {0, 15, 30, 45}, dont les demi-heures gardées
        # par `parse_rte_frame` (minute % 30 == 0), quel que soit l'instant réel où le test tourne.
        self.now, self.bias_rte, self.span_days = now.floor("15min"), bias_rte, span_days

    def _consumption(self, t: pd.Timestamp) -> float:
        return 50000 + 5000 * np.sin(t.hour / 24 * 6.28)

    def get(self, url, params=None, timeout=None):
        if "opendata.reseaux-energies" in url:
            idx = pd.date_range(self.now - pd.Timedelta(days=self.span_days), self.now + pd.Timedelta(days=2), freq="15min")
            rows = ["perimetre;nature;date_heure;consommation;prevision_j1;prevision_j"]
            # RTE publie la prévision J-1 pour l'ensemble du jour local D+1, pas demi-heure par demi-heure
            known_until = self.now.tz_convert(TZ).normalize() + pd.Timedelta(days=1)
            for t in idx:
                c = self._consumption(t)
                measured = f"{c:.1f}" if t <= self.now else ""          # le réel n'existe pas encore dans le futur
                p1 = f"{c - self.bias_rte:.1f}" if t.tz_convert(TZ).normalize() <= known_until else ""
                rows.append(f"France;Données temps réel;{t.strftime('%Y-%m-%dT%H:%M:%S+00:00')};{measured};{p1};{measured}")
            return FakeResponse(text="\n".join(rows))
        # Fenêtre météo généreuse et indépendante de l'horloge réelle de la machine (le code interrogé calcule
        # `forecast_days` à partir de `pd.Timestamp.now()` réel, hors de portée de `self.now` en test).
        idx = pd.date_range((self.now - pd.Timedelta(days=self.span_days)).floor("D"), (self.now + pd.Timedelta(days=16)).ceil("D"), freq="h")
        h = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in idx]}
        for v, val in zip(params["hourly"].split(","), [10.0, 70.0, 10.0, 50.0, 100.0], strict=True):
            h[v] = [val + np.sin(i / 24 * 6.28) for i in range(len(idx))]
        return FakeResponse(payload={"hourly": h})


@pytest.fixture()
def fixed_now():
    return pd.Timestamp("2024-03-19 14:00", tz="Europe/Paris").tz_convert("UTC")


@pytest.fixture()
def provider(fixed_now):
    return LiveProvider(session=FakeSession(fixed_now))


def test_record_day_writes_month_file(tmp_path, trained, provider):
    bundle, _ = trained
    rows = journal.record_day(bundle, provider, "2024-03-20", tmp_path)
    assert len(rows) == 48 and rows["date"].eq("2024-03-20").all()
    assert rows["actual_mw"].isna().all() and rows["rte_j1_mw"].isna().all()
    assert (rows["lower_mw"] <= rows["forecast_mw"]).all() and (rows["forecast_mw"] <= rows["upper_mw"]).all()

    saved = journal._load_month(tmp_path, pd.Period("2024-03", "M"))
    assert len(saved) == 48
    df = journal.load_journal(tmp_path)
    assert len(df) == 48 and list(df.columns) == journal.JOURNAL_COLUMNS


def test_record_day_is_idempotent_and_preserves_reconciled_data(tmp_path, trained, provider, fixed_now):
    bundle, _ = trained
    journal.record_day(bundle, provider, "2024-03-19", tmp_path)      # jour partiellement écoulé (now = 13 h)
    journal.reconcile(provider, tmp_path, lookback_days=5, now=fixed_now)
    before = journal.load_journal(tmp_path)
    assert before["actual_mw"].notna().any()          # au moins le passé récent du jour est déjà connu
    assert before["actual_mw"].isna().any()            # mais pas la fin de journée, pas encore écoulée

    journal.record_day(bundle, provider, "2024-03-19", tmp_path)   # nouvel enregistrement du même jour
    after = journal.load_journal(tmp_path)
    assert len(after) == 48                                          # pas de doublon
    pd.testing.assert_series_equal(before["actual_mw"], after["actual_mw"])   # le réel déjà connu n'est pas effacé


def test_reconcile_fills_actual_and_rte(tmp_path, trained, provider, fixed_now):
    bundle, _ = trained
    journal.record_day(bundle, provider, "2024-03-18", tmp_path)     # jour entièrement passé
    r = journal.reconcile(provider, tmp_path, lookback_days=5, now=fixed_now)
    assert r["months_updated"] == ["2024-03"] and r["rows_filled_actual"] == 48 and r["rows_filled_rte"] == 48

    df = journal.load_journal(tmp_path)
    expected = 50000 + 5000 * np.sin(df["time"].dt.hour / 24 * 6.28)     # formule de FakeSession : heure UTC
    np.testing.assert_allclose(df["actual_mw"].to_numpy(), expected.to_numpy(), atol=1.0)
    np.testing.assert_allclose(df["rte_j1_mw"].to_numpy(), (expected - 300.0).to_numpy(), atol=1.0)

    again = journal.reconcile(provider, tmp_path, lookback_days=5, now=fixed_now)
    assert again["rows_filled_actual"] == 0 and again["rows_filled_rte"] == 0     # déjà complet : rien à refaire


def test_reconcile_only_fills_known_rte_forecast(tmp_path, trained, provider, fixed_now):
    """La prévision RTE J-1 d'un jour encore loin dans le futur n'est pas encore publiée : elle reste NaN."""
    bundle, _ = trained
    journal.record_day(bundle, provider, "2024-03-21", tmp_path)   # jour D+2 : pas encore publié dans le FakeSession
    journal.reconcile(provider, tmp_path, lookback_days=10, now=fixed_now)
    df = journal.load_journal(tmp_path)
    assert df["rte_j1_mw"].isna().all() and df["actual_mw"].isna().all()   # jour futur : ni réel ni prévision RTE


def test_reconcile_respects_lookback(tmp_path, trained, fixed_now):
    bundle, _ = trained
    old_provider = LiveProvider(session=FakeSession(fixed_now, span_days=90))
    journal.record_day(bundle, old_provider, "2024-02-15", tmp_path)   # mois différent de celui du lookback
    journal.reconcile(old_provider, tmp_path, lookback_days=5, now=fixed_now)
    df = journal.load_journal(tmp_path)
    assert df["actual_mw"].isna().all()                                # février non parcouru (lookback = mars)


def test_summary_metrics():
    idx = pd.date_range("2024-03-01", periods=4, freq="30min", tz="UTC")
    df = pd.DataFrame({
        "date": "2024-03-01", "time": idx,
        "forecast_mw": [100.0, 200.0, 300.0, 400.0], "lower_mw": [80.0, 180.0, 280.0, 380.0], "upper_mw": [120.0, 220.0, 320.0, 420.0],
        "actual_mw": [110.0, 190.0, np.nan, 405.0], "rte_j1_mw": [90.0, 210.0, np.nan, 430.0],
        "model_version": "test", "recorded_at": pd.Timestamp.now(tz="UTC"),
    })
    s = journal.summary(df)
    assert set(s.index) == {"Modèle (journal, en direct)", "RTE J-1"}
    assert s.loc["Modèle (journal, en direct)", "n"] == 3            # la ligne sans réel est ignorée
    assert s.loc["Modèle (journal, en direct)", "couverture intervalle %"] == pytest.approx(100.0)


def test_summary_empty():
    assert journal.summary(pd.DataFrame(columns=journal.JOURNAL_COLUMNS)).empty


def test_cli_record_reconcile_summary(tmp_path, monkeypatch, trained, capsys):
    """La commande CLI utilise l'horloge réelle : on enregistre un jour d'hier par rapport à « maintenant »."""
    from conso.model import save_bundle

    bundle, _ = trained
    real_now = pd.Timestamp.now(tz="UTC")
    live_provider = LiveProvider(session=FakeSession(real_now))
    model_dir = tmp_path / "model"
    save_bundle(bundle, model_dir)
    monkeypatch.setattr(journal, "LiveProvider", lambda *a, **k: live_provider)

    import argparse

    yesterday = (real_now.tz_convert("Europe/Paris") - pd.Timedelta(days=1)).date()
    ns = argparse.Namespace(model=str(model_dir), out=str(tmp_path / "j"), date=str(yesterday))
    journal._cli_record(ns)
    out = capsys.readouterr().out
    assert "points enregistrés" in out

    journal._cli_reconcile(argparse.Namespace(out=str(tmp_path / "j"), lookback_days=5))
    out = capsys.readouterr().out
    assert "réel :" in out and "0 points complétés" not in out

    journal._cli_summary(argparse.Namespace(out=str(tmp_path / "j")))
    out = capsys.readouterr().out
    assert "Modèle (journal, en direct)" in out
