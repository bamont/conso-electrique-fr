"""Outils de temps : retards en heure d'horloge locale, masques, bornes de jour."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TZ


def shift_local(s: pd.Series, days: int) -> pd.Series:
    """Valeur de ``s`` à (t_local - ``days`` jours), indexée sur l'index de ``s`` (UTC).

    Le décalage se fait en heure d'horloge : le lundi 8 h est comparé au lundi 8 h précédent,
    même si un changement d'heure a eu lieu entre-temps.
    """
    loc = s.index.tz_convert(TZ).tz_localize(None)
    src = (loc - pd.Timedelta(days=days)).tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward")
    return pd.Series(s.reindex(src.tz_convert("UTC")).to_numpy(), index=s.index, name=s.name)


def mask_between(index: pd.DatetimeIndex, start, end) -> np.ndarray:
    """Masque des horodatages dans [start, end[."""
    return np.asarray((index >= start) & (index < end))


def local_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Date locale (naïve, minuit) de chaque horodatage."""
    return index.tz_convert(TZ).tz_localize(None).normalize()


def day_bounds(date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[début, fin[ du jour local ``date`` (tient compte des jours à 46 ou 50 demi-heures)."""
    d = pd.Timestamp(date).normalize()
    if d.tzinfo is not None:
        d = d.tz_localize(None)
    return d.tz_localize(TZ), (d + pd.Timedelta(days=1)).tz_localize(TZ)
