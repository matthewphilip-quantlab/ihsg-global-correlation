import pandas as pd
import pytest

from src import data_loader as dl
from tests.conftest import make_ohlc, random_walk


@pytest.fixture
def fake_yahoo(monkeypatch):
    """Ganti yf.download dengan data sintetis (format MultiIndex seperti yfinance baru)."""
    calls = []
    data = {"^JKSE": make_ohlc(random_walk(start="2007-01-01", end="2012-12-31"))}

    def fake_download(ticker, start=None, end=None, **kwargs):
        calls.append((ticker, start, end))
        df = data.get(ticker)
        if df is None:
            return pd.DataFrame()
        sub = df.loc[start:(pd.Timestamp(end) - pd.Timedelta(days=1)) if end else None]
        sub = sub.copy()
        sub.columns = pd.MultiIndex.from_product([sub.columns, [ticker]], names=["Price", "Ticker"])
        return sub

    monkeypatch.setattr(dl.yf, "download", fake_download)
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    return calls


def test_download_then_cache(fake_yahoo, tmp_path):
    out = dl.load_prices(["^JKSE"], "2007-01-01", cache_dir=tmp_path, verbose=False)
    assert list(out["^JKSE"].columns) == dl.OHLCV
    assert dl.cache_path("^JKSE", tmp_path).exists()
    n_calls = len(fake_yahoo)
    dl.load_prices(["^JKSE"], "2007-01-01", cache_dir=tmp_path, verbose=False)
    assert len(fake_yahoo) == n_calls                     # kedua kali dari cache


def test_longer_period_or_refresh_triggers_download(fake_yahoo, tmp_path):
    dl.load_prices(["^JKSE"], "2010-01-01", cache_dir=tmp_path, verbose=False)
    n = len(fake_yahoo)
    dl.load_prices(["^JKSE"], "2007-01-01", cache_dir=tmp_path, verbose=False)   # cache terlalu pendek
    assert len(fake_yahoo) == n + 1
    dl.load_prices(["^JKSE"], "2007-01-01", cache_dir=tmp_path, verbose=False, refresh=True)
    assert len(fake_yahoo) == n + 2


def test_failed_ticker_is_empty_and_not_cached(fake_yahoo, tmp_path):
    out = dl.load_prices(["^SET.BK"], "2007-01-01", cache_dir=tmp_path, verbose=False)
    assert out["^SET.BK"].empty
    assert not dl.cache_path("^SET.BK", tmp_path).exists()


def test_close_panel_uses_ids(fake_yahoo, tmp_path):
    prices = dl.load_prices(["^JKSE", "^SET.BK"], "2007-01-01", cache_dir=tmp_path, verbose=False)
    panel = dl.close_panel(prices, {"^JKSE": "JKSE", "^SET.BK": "SET"})
    assert list(panel.columns) == ["JKSE", "SET"] and panel["SET"].isna().all()
