import numpy as np
import pandas as pd

from conso.backtest import rolling_backtest
from conso.config import TZ


def test_rolling_backtest_modes(dataset, wt_fc):
    out = rolling_backtest(
        dataset, wt_fc, pd.Timestamp("2024-02-01", tz=TZ), pd.Timestamp("2024-04-01", tz=TZ), train_start="2022-01-01",
        rounds=60, params={"num_leaves": 15, "min_data_in_leaf": 50, "learning_rate": 0.2}, bias_months=12, progress=False,
    )
    assert list(out.columns) == ["oracle", "fc", "fc_debiased"]
    assert out.index.min() >= pd.Timestamp("2024-02-01", tz=TZ) and out.index.max() < pd.Timestamp("2024-04-01", tz=TZ)
    y = dataset["conso"].loc[out.index]
    mae = {c: float((out[c] - y).abs().mean()) for c in out}
    bias = {c: float((out[c] - y).mean()) for c in out}
    # la météo prévue est biaisée (~ +0,7 °C) : la correction horaire doit réduire l'erreur et le biais
    assert mae["fc_debiased"] < mae["fc"]
    assert abs(bias["fc_debiased"]) < abs(bias["fc"])
    assert mae["oracle"] <= mae["fc_debiased"] * 1.15 # l'oracle reste la borne haute
    assert np.isfinite(out.to_numpy()).all()
