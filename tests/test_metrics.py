import numpy as np
import pandas as pd

from conso.metrics import mase_scale, metrics

def test_metrics_values():
    y = pd.Series([100.0, 200.0, 300.0])
    p = pd.Series([110.0, 190.0, np.nan])
    m = metrics(y, p, scale=10.0)
    assert m["n"] == 2 # les prévisions manquantes sont ignorées
    assert m["MAE"] == 10.0 and m["biais"] == 0.0 and m["MASE"] == 1.0
    assert np.isclose(m["MAPE %"], (10 / 100 + 10 / 200) / 2 * 100)


def test_mase_scale():
    y = pd.Series([1.0, 2.0, 4.0, 8.0])
    lag = pd.Series([0.0, 1.0, 3.0, 6.0])
    assert mase_scale(y, lag, np.array([True, True, True, True])) == np.mean([1, 1, 1, 2])
