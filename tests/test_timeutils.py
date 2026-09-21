import numpy as np
import pandas as pd

from conso.config import TZ
from conso.timeutils import day_bounds, mask_between, shift_local


def _weekly_pattern(start, end):
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="30min")
    loc = idx.tz_convert(TZ)
    return pd.Series(loc.hour * 2 + loc.minute // 30 + 100 * loc.dayofweek, index=idx, dtype=float)


def test_shift_local_follows_wall_clock_across_dst():
    y = _weekly_pattern("2024-03-01", "2024-04-30") # contient le passage à l'heure d'été
    shifted = shift_local(y, 7)
    valid = shifted.notna() & (y.index >= y.index[0] + pd.Timedelta(days=8))
    mismatches = (shifted[valid] != y[valid]).sum()
    naive_utc = (y - y.shift(336))[y.index >= y.index[0] + pd.Timedelta(days=8)]
    assert mismatches <= 4 # seules les heures inexistantes diffèrent
    assert (naive_utc != 0).sum() > 100 # un retard de 336 pas UTC décale d'une heure


def test_day_bounds_handles_dst_days():
    s, e = day_bounds("2024-03-31")
    assert (e - s) == pd.Timedelta(hours=23) # 46 demi-heures
    s, e = day_bounds("2024-10-27")
    assert (e - s) == pd.Timedelta(hours=25) # 50 demi-heures
    s, e = day_bounds("2024-06-15")
    assert (e - s) == pd.Timedelta(hours=24)


def test_mask_between_is_half_open():
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    m = mask_between(idx, idx[1], idx[3])
    assert np.array_equal(m, [False, True, True, False, False])
