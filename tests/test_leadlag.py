import math

import numpy as np
import pandas as pd
import pytest

from src import leadlag as ll
from tests.test_analysis import _daily_pairs


SERIES = pd.DataFrame({"id": ["SYNC", "AFTER"], "name": ["s", "a"], "session": ["sync", "after"]})


# ---- kebersihan data -------------------------------------------------------------------------
def test_in_sample_and_drop_flagged():
    idx = pd.bdate_range("2019-12-25", "2020-01-10")
    df = pd.DataFrame({"X": np.arange(len(idx), dtype=float), "Y": 1.0}, index=idx)
    assert ll.in_sample(df, "2019-12-31").index.max() <= pd.Timestamp("2019-12-31")
    flags = pd.DataFrame({"X": [False, True] + [False] * (len(idx) - 2)}, index=idx)
    out = ll.drop_flagged(df, flags)
    assert np.isnan(out["X"].iloc[1]) and out["Y"].notna().all() and df["X"].notna().all()


# ---- cross-correlation & regresi lead ---------------------------------------------------------------
def test_ccf_peaks_after_at_plus_one_sync_at_zero():
    daily, _ = _daily_pairs()
    ccf = ll.cross_correlation(daily, ["SYNC", "AFTER"], 5)
    assert len(ccf) == 2 * 11 and (ccf["p_bh"] >= ccf["p"] - 1e-12).all()
    peaks = ll.ccf_peaks(ccf, SERIES)
    assert peaks.loc["AFTER", "k_puncak"] == 1 and peaks.loc["SYNC", "k_puncak"] == 0
    assert abs(peaks.loc["AFTER", "r_k0"]) < 0.1 < peaks.loc["AFTER", "r_k1"]


def test_pair_lead_betas_recover_lagged_effect():
    daily, _ = _daily_pairs()
    b = ll.pair_lead_betas(daily, ["SYNC", "AFTER"])
    assert b.loc["AFTER", "ci_lo"] < 0.6 < b.loc["AFTER", "ci_hi"]
    assert b.loc["SYNC", "ci_lo"] < 0 < b.loc["SYNC", "ci_hi"]           # efek sinkron tidak muncul di t+1


def test_grid_returns_and_lead_frame():
    idx = pd.bdate_range("2020-01-01", periods=5)
    close = pd.DataFrame({"T": [100, 101, 102, 103, 104.0], "X": [10, np.nan, 12, 13, 14.0]}, index=idx)
    g = ll.grid_returns(close, ["T", "X"], {})
    assert list(g.index) == [idx[2], idx[3], idx[4]] and np.isclose(g.loc[idx[2], "T"], np.log(102 / 100))
    lf = ll.lead_frame(g, "T")
    assert np.isclose(lf.loc[idx[2], "target_lead"], g.loc[idx[3], "T"]) and len(lf) == 2


# ---- Granger --------------------------------------------------------------------------------------
def test_granger_detects_direction():
    daily, _ = _daily_pairs()
    g = ll.granger_table(daily, ["SYNC", "AFTER"], maxlags=5)
    assert g.loc["AFTER", "p_seri_ke_target_bh"] < 0.001 and g.loc["AFTER", "lag_dipakai"] >= 1
    assert g.loc["AFTER", "p_target_ke_seri"] > 0.01
    assert {"lag_aic", "lag_bic"} <= set(g.columns)


# ---- kointegrasi & EIDO ---------------------------------------------------------------------------
def _coint_pair(n=2000, phi=0.9, seed=0):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 0.01, n))
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = phi * e[t - 1] + rng.normal(0, 0.005)
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.Series(0.2 + x + e, index=idx), pd.Series(x, index=idx)


def test_half_life_of_ar1():
    _, _ = _coint_pair()
    rng = np.random.default_rng(1)
    s = np.zeros(5000)
    for t in range(1, 5000):
        s[t] = 0.8 * s[t - 1] + rng.normal()
    assert ll.half_life(pd.Series(s)) == pytest.approx(-math.log(2) / math.log(0.8), rel=0.1)


def test_cointegration_detected_and_zscore():
    y, x = _coint_pair()
    res = ll.cointegration(y, x)
    assert res["terkointegrasi"] and res["johansen_r0_ditolak"]
    assert res["beta_kointegrasi"] == pytest.approx(1.0, abs=0.05)
    assert res["half_life_hari"] == pytest.approx(-math.log(2) / math.log(0.9), rel=0.25)
    z = ll.spread_zscore(y, x, res)
    assert abs(z.mean()) < 1e-6 and z.std() == pytest.approx(1.0, rel=1e-6)


def test_independent_random_walks_rarely_cointegrated():
    """Ukuran uji: di bawah H0 (dua random walk independen) penolakan ~5%, bukan nol per seed."""
    idx = pd.bdate_range("2010-01-01", periods=1000)
    results = []
    for seed in range(60):
        rng = np.random.default_rng(seed)
        y = pd.Series(np.cumsum(rng.normal(0, 0.01, 1000)), index=idx)
        x = pd.Series(np.cumsum(rng.normal(0, 0.01, 1000)), index=idx)
        results.append(ll.cointegration(y, x))
    rate = np.mean([r["terkointegrasi"] for r in results])
    assert rate <= 0.12
    assert all("half_life_hari" not in r for r in results if not r["terkointegrasi"])


def test_eido_frame_timing_and_identities():
    idx = pd.bdate_range("2015-01-01", periods=6)
    close = pd.DataFrame({"T": [5000, 5050, 5000, 5100, 5150, 5200.0], "FX": [12500, 12600, 12550, 12500, 12450, 12400.0],
                          "E": [27, 27.5, 27.2, 27.8, 28.1, 28.0], "S": [2000, 2010, 2005, 2020, 2030, 2025.0]}, index=idx)
    f = ll.eido_frame(close, "T", "FX", "E", ["S"], {})
    assert list(f.index) == list(idx[1:-1])
    t = idx[2]
    assert np.isclose(f.loc[t, "premium"], np.log(27.2) - np.log(5000 / 12550))
    assert np.isclose(f.loc[t, "d_premium"], f.loc[t, "r_eido"] - f.loc[t, "r_target_usd"])
    assert np.isclose(f.loc[t, "target_lead"], np.log(5100 / 5000)) and "r_S" in f


def test_nested_regressions_incremental_test():
    rng = np.random.default_rng(4)
    n = 2000
    a, b = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame({"y": 0.3 * a + 0.2 * b + rng.normal(size=n), "a": a, "b": b})
    coef, models = ll.nested_regressions(df, "y", {"base": ["a"], "plus_b": ["a", "b"], "b_saja": ["b"]}, "base")
    assert models.loc["plus_b", "p_tambahan"] < 1e-6 and models.loc["plus_b", "tambahan_r2_vs_base"] > 0.02
    assert np.isnan(models.loc["b_saja", "p_tambahan"]) and set(coef["model"]) == {"base", "plus_b", "b_saja"}


# ---- gap pembukaan (eksploratif) ------------------------------------------------------------------
def test_open_gap_split_drops_prev_close_fill():
    idx = pd.bdate_range("2015-01-01", periods=4)
    ohlc = pd.DataFrame({"Open": [100, 101, 103, 104.0], "Close": [100, 103, 104, 102.0]}, index=idx)
    ohlc.loc[idx[2], "Open"] = 103.0                         # Open = Close kemarin -> dibuang
    s = ll.open_gap_split(ohlc)
    assert np.isnan(s.loc[idx[2], "gap"]) and np.isnan(s.loc[idx[2], "intraday"])
    assert np.isclose(s.loc[idx[1], "gap"] + s.loc[idx[1], "intraday"], np.log(103 / 100))


def test_gap_lead_corr_finds_overnight_effect():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2012-01-02", periods=1500)
    x = rng.normal(0, 0.01, len(idx))
    gap = np.concatenate([[0], 0.5 * x[:-1]]) + rng.normal(0, 0.003, len(idx))
    daily = pd.DataFrame({"Date": idx, "id": "X", "r_x": x, "r_target": rng.normal(0, 0.01, len(idx))})
    split = pd.DataFrame({"gap": gap, "intraday": rng.normal(0, 0.01, len(idx))}, index=idx)
    res = ll.gap_lead_corr(daily, split, ["X"]).set_index("bagian")
    assert res.loc["gap", "r"] > 0.7 and abs(res.loc["intraday", "r"]) < 0.1


def test_phase4_summary_runs():
    daily, _ = _daily_pairs()
    ccf = ll.cross_correlation(daily, ["SYNC", "AFTER"], 2)
    peaks = ll.ccf_peaks(ccf, SERIES)
    lead = ll.pair_lead_betas(daily, ["AFTER"])
    g = ll.granger_table(daily, ["AFTER"], 3)
    y, x = _coint_pair()
    co = ll.cointegration(y, x)
    models = pd.DataFrame({"tambahan_r2_vs_base": [0.01], "p_tambahan": [0.02]}, index=["m"])
    txt = ll.phase4_summary(peaks, lead, {"r2": 0.05}, g, co, models, {"AFTER": "After", "SYNC": "Sync"}, "FX",
                            lead_1f=lead.assign(beta=-lead["beta"]))
    assert "k = +1: After" in txt and "half-life" in txt and "tanda berbalik" in txt
    assert np.isfinite(co["half_life_hari_ct_info"])
