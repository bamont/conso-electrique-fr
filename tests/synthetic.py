"""Jeux de données synthétiques pour les tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from conso.calendars import attach_calendar
from conso.config import CITY_WEIGHTS, HOURLY, TZ, WCOLS
from conso.timeutils import local_days


def make_dataset(start: str = "2021-01-01", end: str = "2024-04-30", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23, minutes=30), freq="30min")
    loc = idx.tz_convert(TZ)
    doy, hh, dow = loc.dayofyear.to_numpy(), loc.hour.to_numpy() + loc.minute.to_numpy() / 60, loc.dayofweek.to_numpy()
    day = local_days(idx)

    n_days = (day.max() - day.min()).days + 1
    ar = np.zeros(n_days)
    for i in range(1, n_days):
        ar[i] = 0.8 * ar[i - 1] + rng.normal(0, 2)
    tday = pd.Series(ar, index=pd.date_range(day.min(), periods=n_days))
    temp = 12 - 9 * np.cos(2 * np.pi * (doy - 15) / 365) + 3 * np.sin(2 * np.pi * (hh - 9) / 24) + tday.reindex(day).to_numpy()

    d = pd.DataFrame(index=idx)
    d.index.name = "date_heure"
    d["temp_nat"] = temp
    d["hum_nat"] = 70 + rng.normal(0, 3, len(idx))
    d["wind_nat"] = 10 + rng.normal(0, 1, len(idx))
    d["cloud_nat"] = 50 + rng.normal(0, 5, len(idx))
    d["rad_nat"] = np.clip(300 * np.sin(np.pi * (hh - 6) / 12), 0, None) * (1 - 0.3 * (doy > 300))
    d = attach_calendar(d)

    years = (idx - idx[0]).days.to_numpy() / 365.25
    hdd = np.clip(17 - temp, 0, None)
    d["conso"] = (
        52000 - 700 * years + 1800 * hdd + 4500 * np.sin(2 * np.pi * (hh - 6) / 24) ** 2
        - 4500 * (dow >= 5) - 5000 * d["is_holiday"].to_numpy() - 400 * d["n_zones_vac"].to_numpy()
        + rng.normal(0, 500, len(idx))
    )
    d["source"] = "cons_def"
    d["prevision_j1"] = d["conso"] * (1 + rng.normal(0, 0.012, len(idx))) - 500
    return d


def make_city_forecast(df: pd.DataFrame, start: str = "2022-01-01", seed: int = 1) -> pd.DataFrame:
    """Prévisions météo horaires par ville : observé + biais de température par heure + bruit."""
    rng = np.random.default_rng(seed)
    hourly = df[WCOLS].resample("1h").mean().loc[start:]
    hours = hourly.index.tz_convert(TZ).hour.to_numpy()
    bias = 0.7 + 0.4 * np.cos(2 * np.pi * (hours - 22) / 24)          # ~ +1,1 °C la nuit, ~ +0,3 °C à midi
    frames = []
    for i, city in enumerate(CITY_WEIGHTS):
        f = pd.DataFrame(index=hourly.index)
        f["temperature_2m"] = hourly["temp_nat"] + bias + rng.normal(0, 0.8, len(f)) + 0.03 * i
        f["relative_humidity_2m"] = hourly["hum_nat"] + rng.normal(0, 3, len(f))
        f["wind_speed_10m"] = hourly["wind_nat"] + rng.normal(0, 1, len(f))
        f["cloud_cover"] = hourly["cloud_nat"] + rng.normal(0, 5, len(f))
        f["shortwave_radiation"] = hourly["rad_nat"] + rng.normal(0, 15, len(f))
        f["city"] = city
        f.index.name = "time"
        frames.append(f[[*HOURLY, "city"]])
    return pd.concat(frames).sort_index()
