"""Calendrier : jours fériés, ponts et vacances scolaires par zone (A, B, C)."""

from __future__ import annotations

import holidays
import pandas as pd
from vacances_scolaires_france import SchoolHolidayDates

from .timeutils import local_days

CALENDAR_COLUMNS = ["is_holiday", "is_bridge", "vac_A", "vac_B", "vac_C", "n_zones_vac"]


def calendar_frame(dates) -> pd.DataFrame:
    """Variables de calendrier pour des dates locales (une ligne par date)."""
    dates = pd.DatetimeIndex(dates).normalize()
    fr = holidays.country_holidays("FR", years=range(dates.min().year - 1, dates.max().year + 2))
    shd = SchoolHolidayDates()

    def is_h(d):
        return d.date() in fr

    out = pd.DataFrame(index=dates)
    out["is_holiday"] = [is_h(d) for d in dates]
    one = pd.Timedelta(days=1)
    # pont : lundi si le mardi est férié, vendredi si le jeudi est férié
    out["is_bridge"] = [(d.dayofweek == 0 and is_h(d + one)) or (d.dayofweek == 4 and is_h(d - one)) for d in dates]
    for z in "ABC":
        out[f"vac_{z}"] = [float(shd.is_holiday_for_zone(d.date(), z)) for d in dates]
    out["n_zones_vac"] = out[["vac_A", "vac_B", "vac_C"]].sum(axis=1)
    return out


def attach_calendar(d: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes de calendrier à un DataFrame indexé en UTC (calcul en heure locale)."""
    days = local_days(d.index)
    frame = calendar_frame(days.unique())
    out = d.copy()
    for c in CALENDAR_COLUMNS:
        out[c] = frame[c].reindex(days).to_numpy()
    out["is_holiday"] = out["is_holiday"].astype(bool)
    out["is_bridge"] = out["is_bridge"].astype(bool)
    return out
