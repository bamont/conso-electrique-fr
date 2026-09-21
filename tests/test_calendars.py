import pandas as pd

from conso.calendars import attach_calendar, calendar_frame


def test_holidays_and_bridges():
    f = calendar_frame(pd.date_range("2025-04-28", "2025-05-04"))
    assert f.loc["2025-05-01", "is_holiday"] # fête du Travail (jeudi)
    assert f.loc["2025-05-02", "is_bridge"] # vendredi après un jeudi férié
    assert not f.loc["2025-04-30", "is_bridge"]
    g = calendar_frame(pd.date_range("2022-10-28", "2022-11-02"))
    assert g.loc["2022-11-01", "is_holiday"] # Toussaint (mardi)
    assert g.loc["2022-10-31", "is_bridge"] # lundi avant un mardi férié


def test_school_holidays_toussaint_2024():
    f = calendar_frame(pd.to_datetime(["2024-10-25", "2024-11-05"]))
    assert f.loc["2024-10-25", "n_zones_vac"] == 3
    assert f.loc["2024-11-05", "n_zones_vac"] == 0


def test_attach_calendar_uses_local_dates():
    idx = pd.date_range("2025-04-30 21:00", periods=6, freq="h", tz="UTC") # 23 h locales le 30/04 -> 1er mai
    d = attach_calendar(pd.DataFrame({"x": range(6)}, index=idx))
    assert d["is_holiday"].dtype == bool
    assert list(d["is_holiday"]) == [False, True, True, True, True, True] # 22 h UTC = minuit local (UTC+2)
