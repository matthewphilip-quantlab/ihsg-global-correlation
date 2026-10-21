import json

import numpy as np
import pandas as pd
import pytest

from src import preprocess as pp
from src.config import series_table
from tests.conftest import make_ohlc, random_walk


def _fx_2013_like():
    """Pola Nov-Des 2013: run salah 3 hari, 2 hari harga asli, run salah 2 hari (nilai basi konstan)."""
    s = random_walk(start="2013-09-02", end="2014-01-31", sd=0.002, level=11800)
    run3 = pd.bdate_range("2013-11-29", "2013-12-03")      # Jum, Sen, Sel
    run2 = pd.bdate_range("2013-12-06", "2013-12-09")      # Jum, Sen
    s.loc[run3] = 9612.45
    s.loc[run2] = 9612.45
    return s, run3, run2, pd.bdate_range("2013-12-04", "2013-12-05")


# ---- bad tick FX (Amandemen 1) ----------------------------------------------------------
def test_three_day_run_caught_with_amendment(qcfg):
    s, run3, run2, good = _fx_2013_like()
    bad = pp.fx_bad_ticks(s, qcfg["spike_threshold"]["fx"], qcfg["reversal_ratio"],
                          qcfg["fx_cleaning"]["max_reversal_days"])
    assert set(bad.index) == set(run3) | set(run2)
    assert bad.loc[run3[0], "run_days"] == 3 and bad.loc[run2[0], "run_days"] == 2


def test_regression_two_day_limit_removes_good_prices(qcfg):
    """Batas 2 hari keliru: harga asli 4-5 Des terbuang, run 3 hari lolos. Amandemen 1 tidak mengulanginya."""
    s, run3, run2, good = _fx_2013_like()
    k2 = pp.fx_bad_ticks(s, 0.05, 0.5, max_days=2)
    assert set(good) <= set(k2.index) and not set(run3) & set(k2.index)
    k5 = pp.fx_bad_ticks(s, 0.05, 0.5, qcfg["fx_cleaning"]["max_reversal_days"])
    assert not set(good) & set(k5.index)


def test_old_rule_only_catches_single_day(qcfg):
    s, run3, run2, _ = _fx_2013_like()
    one = s.index[20]
    s.loc[one] *= 0.1                                    # harga kurang satu digit
    old = pp.fx_bad_ticks(s, 0.05, 0.5, qcfg["fx_cleaning"]["max_reversal_days_old"])
    assert list(old.index) == [one]


def test_persistent_move_and_weak_reversal_not_removed():
    s = random_walk(sd=0.001, level=10000)
    s.iloc[300:] *= 1.08                                 # devaluasi permanen
    s.iloc[600] *= 1.07                                  # lonjakan yang hanya dibalik ~29% besoknya
    s.iloc[601:] *= 1.05
    assert pp.fx_bad_ticks(s, 0.05, 0.5, 5).empty


# ---- hari libur & kuotasi basi -----------------------------------------------------------
def test_holiday_rows_suspend_and_stale(cfg):
    df = make_ohlc(random_walk(start="2008-09-01", end="2008-12-31", level=1500))
    susp = cfg["sanity_checks"]["jkse_2008"]["suspended_days"]
    stale_day, real_flat = df.index[5], df.index[10]
    df.loc[stale_day, "Close"] = df["Close"].iloc[4]
    df.loc[stale_day, "Volume"] = 0                      # libur diisi harga kemarin
    df.loc[real_flat, "Close"] = df["Close"].iloc[9]     # close sama tapi ada volume -> tetap hari bursa
    rows = pp.holiday_rows(df, susp)
    assert set(rows.index) == {stale_day} | set(pd.to_datetime(susp))


def test_stale_quote_flags_repeated_value_only():
    s = random_walk(start="2012-01-02", end="2013-12-31", sd=0.003, level=9600)
    mondays = s.index[s.index.dayofweek == 0][::4][:12]
    s.loc[mondays] = 9612.45
    flags = pp.stale_quote_flags(s, min_count=10, tol_pct=0.05)
    assert set(mondays) <= set(flags[flags].index)
    assert flags.sum() < len(mondays) + 10               # hanya sekitar nilai berulang
    assert not pp.stale_quote_flags(s, min_count=20, tol_pct=0.05).any()


# ---- panel bersih ------------------------------------------------------------------------
def _toy_prices(cfg):
    s_fx, run3, run2, _ = _fx_2013_like()
    idx = s_fx.index
    prices = {
        "^JKSE": make_ohlc(random_walk(start=idx[0], end=idx[-1], level=4300, seed=2)),
        "IDR=X": make_ohlc(s_fx),
        "CNY=X": make_ohlc(s_fx / 1800),                 # seri fx kedua: aturan generik
        "^VIX": make_ohlc(random_walk(start=idx[0], end=idx[-1], level=15, seed=3)),
    }
    prices["^VIX"].loc[idx[30], "Close"] *= 2            # gerakan besar non-FX: tidak boleh dibuang
    table = series_table({
        "target": [dict(id="JKSE", ticker="^JKSE", name="IHSG", category="index", session="target", transform="log")],
        "core": [dict(id="USDIDR", ticker="IDR=X", name="USD/IDR", category="fx", session="after", transform="log"),
                 dict(id="USDCNY", ticker="CNY=X", name="USD/CNY", category="fx", session="after", transform="log"),
                 dict(id="VIX", ticker="^VIX", name="VIX", category="sentiment", session="after", transform="log")],
    })
    return prices, table, set(run3) | set(run2)


def test_clean_prices_generic_fx_and_logged(cfg):
    prices, table, bad = _toy_prices(cfg)
    panel, log, flags = pp.clean_prices(prices, table, cfg)
    for sid in ("USDIDR", "USDCNY"):
        assert set(panel[sid][panel[sid].isna()].index) >= bad
        assert set(log.loc[log["id"] == sid, "Date"]) == bad
    assert panel["VIX"].notna().all() and "VIX" not in set(log["id"])
    assert list(log.columns) == pp.LOG_COLS
    assert set(flags.columns) == {"USDIDR", "USDCNY"}


def test_clean_prices_old_rule_keeps_runs(cfg):
    prices, table, bad = _toy_prices(cfg)
    panel, log, _ = pp.clean_prices(prices, table, cfg, fx_max_days=1)
    assert log.empty and panel["USDIDR"].notna().all()


# ---- returns -----------------------------------------------------------------------------
def test_pair_daily_returns_common_dates_without_ffill():
    idx = pd.bdate_range("2020-01-01", periods=6)
    close = pd.DataFrame({"T": [100, 101, np.nan, 103, 104, 105.0],
                          "X": [10, 11, 12, np.nan, 14, 15.0],
                          "Y": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]}, index=idx)
    flags = pd.DataFrame({"X": [False, False, False, False, True, False]}, index=idx)
    d = pp.pair_daily_returns(close, "T", {"T": "log", "X": "log", "Y": "diff"}, flags)
    x = d[d["id"] == "X"].set_index("Date")
    assert list(x.index) == [idx[1], idx[4], idx[5]]     # hanya tanggal yang dua-duanya buka
    assert np.isclose(x.loc[idx[4], "r_target"], np.log(104 / 101))
    assert np.isclose(x.loc[idx[4], "r_x"], np.log(14 / 11))
    assert list(x["flag_stale_quote"]) == [False, True, True]
    y = d[d["id"] == "Y"].set_index("Date")
    assert np.isclose(y.loc[idx[3], "r_x"], 0.2)         # yield/diff memakai selisih, lintas hari libur T
    assert not d["flag_stale_quote"][d["id"] == "Y"].any()


def test_weekly_returns_use_last_available_price():
    idx = pd.bdate_range("2020-01-06", "2020-01-31")
    close = pd.DataFrame({"T": np.arange(len(idx), dtype=float) + 100}, index=idx)
    close.loc["2020-01-17", "T"] = np.nan                # Jumat kosong -> pakai Kamis
    w = pp.weekly_returns(close, {"T": "log"})
    assert np.isclose(w.loc["2020-01-17", "T"], np.log(108 / 104))
    assert np.isclose(w.loc["2020-01-24", "T"], np.log(114 / 108))


def test_jkse_in_usd_common_dates():
    idx = pd.bdate_range("2020-01-01", periods=3)
    close = pd.DataFrame({"J": [6000, np.nan, 6300.0], "F": [15000, 15500, 14000.0]}, index=idx)
    usd = pp.jkse_in_usd(close, "J", "F", "J_USD")
    assert list(usd.index) == [idx[0], idx[2]] and np.isclose(usd.iloc[0], 0.4)


# ---- uji ulang ----------------------------------------------------------------------------
def test_floor3_never_rounds_up():
    assert pp.floor3(0.2999) == 0.299
    assert pp.floor3(-0.30049) == 0.300
    assert pp.floor3(0.1 + 0.2) == 0.300                 # 0.30000000000000004
    assert pp.floor3(0.3 - 1e-12) == 0.300               # galat float tidak menjatuhkan ke 0.299


def test_usdidr_retest_and_decision_flags_disagreement():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2010-01-01", "2019-12-31")
    t = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    x_ret = -0.5 * np.log(t[1:] / t[:-1]) + rng.normal(0, 0.006, len(idx) - 1)
    x = 10000 * np.exp(np.concatenate([[0], np.cumsum(x_ret)]))
    clean = pd.DataFrame({"T": t, "X": x}, index=idx)
    dirty = clean.copy()
    dirty.loc[dirty.index[dirty.index.dayofweek == 4][20:400:40], "X"] *= 0.8  # Jumat salah, terpisah
    tr = {"T": "log", "X": "log"}
    res = pp.usdidr_retest({"lama": (dirty, False), "baru": (clean, True)}, "X", "T", tr,
                           ["2010-01-01", "2019-12-31"], 0.300)
    assert res.loc[1, "lolos"] and not res.loc[0, "lolos"]
    assert res["n_minggu"].between(515, 525).all()
    dec = pp.retest_decision(res, 0.300, "lama")
    assert dec["status"] == "driver" and dec["differs_from_old_rule"] and "PERHATIAN" in dec["note"]


def test_load_phase1_decisions_requires_close_to_close(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"return_mode": "close_to_close"}))
    assert pp.load_phase1_decisions(p)["return_mode"] == "close_to_close"
    p.write_text(json.dumps({"return_mode": "gap_intraday"}))
    with pytest.raises(NotImplementedError):
        pp.load_phase1_decisions(p)


def test_outputs_written(tmp_path, cfg):
    prices, table, _ = _toy_prices(cfg)
    panel, log, flags = pp.clean_prices(prices, table, cfg)
    daily = pp.pair_daily_returns(panel, "JKSE", {"USDIDR": "log", "USDCNY": "log", "VIX": "log"}, flags)
    cov = pp.pair_coverage(daily)
    retest = pd.DataFrame([{"varian": "baru", "r": -0.2, "abs_r_3dp": 0.2, "n_minggu": 520,
                            "lolos": False, "mengikat": True}])
    dec = {"status": "tool_permanen", "reason": "x", "note": "y"}
    paths = pp.save_phase2_outputs(tmp_path, log, retest, flags[flags.any(axis=1)], cov, pd.DataFrame(),
                                   dec, {"n_removed": len(log)})
    assert all(p.exists() for p in paths)
    assert json.loads((tmp_path / "decisions.json").read_text())["usdidr_retest"]["status"] == "tool_permanen"
    assert all(p.exists() for p in pp.save_processed(tmp_path / "proc", {"daily": daily, "s": panel["JKSE"]}))
    text = pp.phase2_summary(log, flags, {"reason": "r", "note": "n"}, cov, pd.DataFrame(), "USDIDR", 0)
    assert "USDIDR" in text and "Phase 2" in text
