"""Sources d'entrées pour l'API : rejeu d'un jour passé (hors ligne) ou données en direct."""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from .calendars import attach_calendar
from .config import HISTORY_DAYS, ODRE_EXPORT, ODRE_PARAMS, TZ, WCOLS
from .data import load_dataset, load_national_forecast
from .features import REQUIRED_COLUMNS
from .timeutils import day_bounds, local_days
from .weather import apply_hourly_bias, fetch_national_forecast, national_30min


@dataclass
class Inputs:
    """Entrées d'une prévision : historique, météo prévue brute, et (en rejeu) réel et prévision RTE."""

    d: pd.DataFrame
    wt_raw: pd.DataFrame
    actual: pd.Series | None = None
    rte_j1: pd.Series | None = None


class ReplayProvider:
    """Rejoue la prévision d'un jour passé à partir du jeu de données et des prévisions météo historiques."""

    def __init__(self, dataset: pd.DataFrame, wt_fc: pd.DataFrame):
        self.df, self.wt_fc = dataset, wt_fc

    @classmethod
    def from_files(cls, dataset_path: str | Path, meteo_fc_path: str | Path) -> ReplayProvider:
        df = load_dataset(dataset_path)
        return cls(df, load_national_forecast(meteo_fc_path, df.index))

    def available_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Premier et dernier jour rejouables : consommation et météo prévue présentes, et assez d'historique."""
        ok = (self.df["conso"].notna() & self.wt_fc["temp_nat"].notna()).to_numpy()
        per_day = pd.Series(1, index=local_days(self.df.index[ok])).groupby(level=0).sum()
        full = per_day[per_day >= 46].index # jours complets (46 à 50 demi-heures)
        if len(full) == 0:
            raise LookupError("Aucun jour rejouable dans le jeu de données.")
        first_history = local_days(self.df.index[:1])[0]
        return max(full.min(), first_history + pd.Timedelta(days=HISTORY_DAYS)), full.max()

    def get_inputs(self, target_date, bias: pd.Series | None = None) -> Inputs:
        start, end = day_bounds(target_date)
        lo, hi = start - pd.Timedelta(days=HISTORY_DAYS), end + pd.Timedelta(days=1)
        sel = (self.df.index >= lo) & (self.df.index < hi)
        d = self.df.loc[sel, REQUIRED_COLUMNS].copy()
        day_rows = (d.index >= start) & (d.index < end)
        if day_rows.sum() == 0:
            raise LookupError(f"Le jeu de données ne couvre pas le {pd.Timestamp(target_date).date()}.")
        if d.index.min() > start - pd.Timedelta(days=HISTORY_DAYS - 1):
            raise LookupError("Historique insuffisant avant cette date.")
        wt_raw = self.wt_fc.loc[sel]
        if wt_raw["temp_nat"].loc[(wt_raw.index >= start) & (wt_raw.index < end)].isna().all():
            raise LookupError("Pas de prévision météo historique pour cette date (couverture depuis 2022).")
        actual = self.df.loc[(self.df.index >= start) & (self.df.index < end), "conso"]
        rte = self.df.loc[actual.index, "prevision_j1"] if "prevision_j1" in self.df.columns else None
        return Inputs(d, wt_raw, actual, rte)


# Données en direct
def _norm_col(c) -> str:
    c = unicodedata.normalize("NFKD", str(c)).encode("ascii", "ignore").decode().lower()
    c = re.sub(r"\(.*?\)", "", c)
    return re.sub(r"[^a-z0-9]+", "_", c).strip("_")


def parse_rte_export(text: str) -> pd.Series:
    """Consommation au pas de 30 min (index UTC) à partir d'un export CSV éCO2mix temps réel."""
    df = pd.read_csv(io.StringIO(text), sep=";")
    if df.shape[1] == 1:
        df = pd.read_csv(io.StringIO(text), sep=",")
    df.columns = [_norm_col(c) for c in df.columns]
    if "date_heure" not in df.columns or "consommation" not in df.columns:
        raise KeyError(f"Colonnes attendues absentes de l'export RTE. Colonnes : {list(df.columns)}")
    if "perimetre" in df.columns:
        m = df["perimetre"].astype(str).str.lower().eq("france")
        if m.any():
            df = df[m]
    df["date_heure"] = pd.to_datetime(df["date_heure"], utc=True)
    df["consommation"] = pd.to_numeric(df["consommation"], errors="coerce")
    s = df.dropna(subset=["consommation"]).set_index("date_heure")["consommation"].sort_index()
    s = s[~s.index.duplicated(keep="first")]
    return s[s.index.minute % 30 == 0]


class LiveProvider:
    """Historique de consommation RTE (temps réel) + météo prévue Open-Meteo, en direct."""

    def __init__(self, session=None):
        self.session = session or requests

    def fetch_consumption(self) -> pd.Series:
        r = self.session.get(ODRE_EXPORT.format("eco2mix-national-tr"), params=ODRE_PARAMS, timeout=300)
        r.raise_for_status()
        return parse_rte_export(r.text)

    def get_inputs(self, target_date, bias: pd.Series | None = None) -> Inputs:
        start, end = day_bounds(target_date)
        lo, hi = start - pd.Timedelta(days=HISTORY_DAYS), end + pd.Timedelta(days=1)
        idx = pd.date_range(lo.tz_convert("UTC"), hi.tz_convert("UTC"), freq="30min", inclusive="left")

        conso = self.fetch_consumption().reindex(idx)
        if conso.loc[idx < start - pd.Timedelta(days=2)].isna().mean() > 0.10:
            raise LookupError("Historique de consommation RTE incomplet : prévision impossible.")

        past = (pd.Timestamp.now(tz="UTC") - lo.tz_convert("UTC")).days + 2
        nat = fetch_national_forecast(past_days=min(past, 92), forecast_days=min(max((end - pd.Timestamp.now(tz=TZ)).days + 2, 2), 16),
                                      session=self.session)
        wt_raw = national_30min(nat, idx)
        if wt_raw["temp_nat"].loc[(idx >= start) & (idx < end)].isna().any():
            raise LookupError("Prévision météo indisponible pour ce jour.")

        d = pd.DataFrame({"conso": conso}, index=idx)
        # météo passée : même produit que les prévisions, corrigé du même biais horaire
        hist = apply_hourly_bias(wt_raw, bias) if bias is not None else wt_raw
        for c in WCOLS:
            d[c] = hist[c]
        d = attach_calendar(d)
        return Inputs(d[REQUIRED_COLUMNS], wt_raw)
