import numpy as np
import pandas as pd
import pytest

from src.config import load_config


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def qcfg(cfg):
    return cfg["quality"]


def make_ohlc(close: pd.Series, gap_sd: float = 0.002, seed: int = 0) -> pd.DataFrame:
    """OHLC sintetis yang konsisten: Open beda dari Close kemarin, High/Low membungkus Open & Close."""
    rng = np.random.default_rng(seed)
    prev = close.shift(1).fillna(close.iloc[0])
    open_ = prev * np.exp(rng.normal(0, gap_sd, len(close)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, len(close))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, len(close))))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                         "Volume": 1e6}, index=close.index)


def random_walk(start="2007-01-01", end="2012-12-31", sd=0.01, level=100.0, seed=1) -> pd.Series:
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    return pd.Series(level * np.exp(np.cumsum(rng.normal(0, sd, len(dates)))), index=dates)
