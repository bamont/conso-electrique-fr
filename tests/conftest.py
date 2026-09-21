import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from conso.model import Bundle  # noqa: E402
from conso.train import train_bundle  # noqa: E402
from conso.weather import national_30min, weighted_national  # noqa: E402
from synthetic import make_city_forecast, make_dataset  # noqa: E402


@pytest.fixture(scope="session")
def dataset():
    return make_dataset()


@pytest.fixture(scope="session")
def city_forecast(dataset):
    return make_city_forecast(dataset)


@pytest.fixture(scope="session")
def wt_fc(dataset, city_forecast):
    return national_30min(weighted_national(city_forecast), dataset.index)


@pytest.fixture(scope="session")
def trained(dataset, wt_fc) -> tuple[Bundle, dict]:
    """Petit modèle entraîné sur les données synthétiques."""
    import pandas as pd

    return train_bundle(
        dataset, wt_fc, train_start="2021-01-01", train_end=pd.Timestamp("2024-04-01", tz="Europe/Paris"),
        rounds=60, q_rounds=40, params={"num_leaves": 15, "min_data_in_leaf": 50, "learning_rate": 0.2},
        calib_months=6, bias_months=12, min_per_hour=50, version="test",
    )
