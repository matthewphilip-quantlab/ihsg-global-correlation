"""Download harga dari Yahoo Finance dengan cache parquet per ticker.

Cache ada di data/raw/ (masuk .gitignore). Setiap ticker disimpan sebagai satu
file parquet berisi OHLCV, plus data/raw/_manifest.json yang mencatat periode
yang diminta dan tanggal download, supaya cache otomatis diperbarui kalau
periode yang diminta lebih panjang atau datanya sudah terlalu lama.
"""
from __future__ import annotations

import json
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from .config import PROJECT_ROOT

CACHE_DIR = PROJECT_ROOT / "data" / "raw"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
MANIFEST = "_manifest.json"


def cache_path(ticker: str, cache_dir: Path = CACHE_DIR) -> Path:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_")
    return Path(cache_dir) / f"{safe}.parquet"


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], name="Date"), dtype=float)


def clean_frame(raw: pd.DataFrame | None) -> pd.DataFrame:
    """Rapikan output yfinance: kolom satu level, index tanggal tanpa timezone, tanpa duplikat."""
    if raw is None or raw.empty:
        return _empty()
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):  # yfinance baru: (Price, Ticker)
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, [c for c in OHLCV if c in df.columns]].astype(float)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(how="all")


def download_one(ticker: str, start: pd.Timestamp, end: pd.Timestamp | None = None,
                 retries: int = 3, pause: float = 3.0) -> pd.DataFrame:
    """Download satu ticker. Retry dengan jeda bertambah (Yahoo kadang rate-limit)."""
    end_exclusive = (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if end is not None else None
    last_err = "data kosong"
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end_exclusive,
                              auto_adjust=True, progress=False, threads=False)
            df = clean_frame(raw)
            if not df.empty:
                return df
            last_err = "data kosong"
        except Exception as exc:  # koneksi / rate limit
            last_err = repr(exc)
        if attempt < retries:
            time.sleep(pause * attempt)
    print(f"  ! {ticker}: gagal download setelah {retries}x ({last_err})")
    return _empty()


def _read_manifest(cache_dir: Path) -> dict:
    path = Path(cache_dir) / MANIFEST
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _write_manifest(cache_dir: Path, manifest: dict) -> None:
    path = Path(cache_dir) / MANIFEST
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _cache_is_fresh(entry: dict | None, start: pd.Timestamp, end: pd.Timestamp | None,
                    max_age_days: int) -> bool:
    if not entry:
        return False
    if pd.Timestamp(entry["start"]) > start:          # cache lebih pendek dari yang diminta
        return False
    wanted_end = end.strftime("%Y-%m-%d") if end is not None else None
    if entry.get("end") != wanted_end:
        return False
    if end is None:                                   # "sampai terbaru" -> cek umur cache
        age = (date.today() - date.fromisoformat(entry["downloaded"])).days
        return age <= max_age_days
    return True


def load_prices(tickers: list[str], start: str | pd.Timestamp, end: str | pd.Timestamp | None = None,
                refresh: bool = False, max_age_days: int = 7, cache_dir: Path = CACHE_DIR,
                verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Ambil OHLCV untuk banyak ticker, pakai cache kalau masih valid.

    Return: dict {ticker: DataFrame OHLCV}. Ticker yang gagal -> DataFrame kosong
    (tidak di-cache, jadi dicoba lagi di run berikutnya).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end is not None else None
    manifest = _read_manifest(cache_dir)

    out: dict[str, pd.DataFrame] = {}
    from_cache, downloaded, failed = [], [], []
    for ticker in tickers:
        path = cache_path(ticker, cache_dir)
        if not refresh and path.exists() and _cache_is_fresh(manifest.get(ticker), start_ts, end_ts, max_age_days):
            df = pd.read_parquet(path)
            from_cache.append(ticker)
        else:
            df = download_one(ticker, start_ts, end_ts)
            if df.empty:
                failed.append(ticker)
            else:
                df.to_parquet(path)
                manifest[ticker] = {
                    "start": start_ts.strftime("%Y-%m-%d"),
                    "end": end_ts.strftime("%Y-%m-%d") if end_ts is not None else None,
                    "downloaded": date.today().isoformat(),
                    "first_date": df.index[0].strftime("%Y-%m-%d"),
                    "last_date": df.index[-1].strftime("%Y-%m-%d"),
                    "rows": int(len(df)),
                }
                downloaded.append(ticker)
        out[ticker] = df.loc[start_ts:end_ts] if not df.empty else df

    _write_manifest(cache_dir, manifest)
    if verbose:
        print(f"Cache: {len(from_cache)} | Download baru: {len(downloaded)} | Gagal: {len(failed)}"
              + (f" -> {', '.join(failed)}" if failed else ""))
    return out


def close_panel(prices: dict[str, pd.DataFrame], id_by_ticker: dict[str, str]) -> pd.DataFrame:
    """Gabungkan kolom Close semua seri jadi satu DataFrame lebar, kolom = id seri."""
    cols = {}
    for ticker, sid in id_by_ticker.items():
        df = prices.get(ticker)
        cols[sid] = df["Close"] if df is not None and not df.empty else pd.Series(dtype=float)
    return pd.DataFrame(cols).sort_index()


def download_actions(ticker: str, start: pd.Timestamp, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Close tanpa penyesuaian dividen (sudah disesuaikan split oleh Yahoo) dan dividen tunai per ex-date."""
    end_exclusive = (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if end is not None else None
    raw = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end_exclusive, auto_adjust=False,
                      actions=True, progress=False, threads=False)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["close_raw", "dividend"], index=pd.DatetimeIndex([], name="Date"), dtype=float)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    idx = pd.DatetimeIndex(raw.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out = pd.DataFrame({"close_raw": raw["Close"].to_numpy(dtype=float),
                        "dividend": raw["Dividends"].to_numpy(dtype=float) if "Dividends" in raw else 0.0},
                       index=idx.normalize())
    out.index.name = "Date"
    return out[~out.index.duplicated(keep="last")].sort_index()


def load_actions(tickers: list[str], start: str | pd.Timestamp, end: str | pd.Timestamp | None = None,
                 refresh: bool = False, max_age_days: int = 7, cache_dir: Path = CACHE_DIR) -> dict[str, pd.DataFrame]:
    """Seperti load_prices, untuk aksi korporasi (cache data/raw/<ticker>_actions.parquet, key manifest '<ticker>#actions')."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end is not None else None
    manifest = _read_manifest(cache_dir)
    out = {}
    for ticker in tickers:
        key = f"{ticker}#actions"
        path = cache_path(ticker, cache_dir).with_name(cache_path(ticker, cache_dir).stem + "_actions.parquet")
        if not refresh and path.exists() and _cache_is_fresh(manifest.get(key), start_ts, end_ts, max_age_days):
            df = pd.read_parquet(path)
        else:
            df = download_actions(ticker, start_ts, end_ts)
            if not df.empty:
                df.to_parquet(path)
                manifest[key] = {"start": start_ts.strftime("%Y-%m-%d"),
                                 "end": end_ts.strftime("%Y-%m-%d") if end_ts is not None else None,
                                 "downloaded": date.today().isoformat(), "rows": int(len(df))}
        out[ticker] = df.loc[start_ts:end_ts] if not df.empty else df
    _write_manifest(cache_dir, manifest)
    return out
