import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import regime as rg


def _weekly_shift(n=700, shift_at=450, temporary=False, seed=0):
    """T vs X: korelasi 0.6 sebelum shift_at, 0.1 sesudahnya (atau kembali 0.6 setelah 100 minggu kalau temporary)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2008-01-04", periods=n, freq="W-FRI")
    x = rng.normal(size=n)
    rho = np.full(n, 0.6)
    rho[shift_at:] = 0.1
    if temporary:
        rho[shift_at + 100:] = 0.6
    t = rho * x + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    return pd.DataFrame({"T": t, "X": x}, index=idx), idx[shift_at]


# ---- rolling -----------------------------------------------------------------------------------
def test_rolling_corr_min_periods_and_shift():
    w, br = _weekly_shift()
    r = rg.rolling_corr(w, "T", ["X"], 52)
    assert r["X"].iloc[:41].isna().all() and r["X"].iloc[45:].notna().all()
    assert r.loc[:br, "X"].iloc[-1] > 0.4 and r["X"].iloc[-1] < 0.35


def test_rolling_shift_summary_trend_vs_spike():
    w, br = _weekly_shift()
    trend = rg.rolling_shift_summary(rg.rolling_corr(w, "T", ["X"], 52), str(br.date()), split_years=2).loc["X"]
    assert trend.rata_pasca_awal < trend.rata_pra - 0.2 and trend.rata_pasca_akhir < trend.rata_pra - 0.2
    w2, br2 = _weekly_shift(temporary=True)
    spike = rg.rolling_shift_summary(rg.rolling_corr(w2, "T", ["X"], 52), str(br2.date()), split_years=3).loc["X"]
    assert spike.rata_pasca_awal < spike.rata_pra - 0.15 and abs(spike.rata_pasca_akhir - spike.rata_pra) < 0.15


# ---- event study --------------------------------------------------------------------------------
def test_episode_window_rolls_forward_and_truncates():
    td = pd.bdate_range("2020-01-01", periods=100)
    w = rg.episode_window(td, "2020-02-01", (-20, 60))        # Sabtu -> hari bursa berikutnya
    assert w.day0 == pd.Timestamp("2020-02-03") and w.base == pd.Timestamp("2020-01-31")
    assert len(w.days) == 81 and not w.truncated
    w2 = rg.episode_window(td, td[90], (-20, 60))
    assert w2.truncated and w2.end == td[-1]
    with pytest.raises(ValueError):
        rg.episode_window(td, "2020-01-03", (-20, 60))


def _crash_path():
    td = pd.bdate_range("2019-01-01", periods=400)
    i0 = 300
    p = np.full(len(td), 100.0)
    p[i0:i0 + 11] = np.linspace(100, 70, 11)                  # jatuh 30% dalam 10 hari bursa
    p[i0 + 11:] = np.linspace(72, 90, len(td) - i0 - 11)
    return td, i0, pd.Series(p, index=td)


def test_episode_metrics_drawdown_trough_cumret():
    td, i0, s = _crash_path()
    w = rg.episode_window(td, td[i0], (-20, 60))
    m = rg.episode_metrics(s, w, td)
    assert m["max_drawdown"] == pytest.approx(-0.30) and m["hari_ke_titik_terendah"] == 10
    assert m["return_ke_titik_terendah"] == pytest.approx(-0.30)
    assert m["return_kumulatif_akhir"] == pytest.approx(s.iloc[i0 + 60] / 100 - 1)


def test_episode_metrics_uses_own_calendar_without_ffill():
    td, i0, s = _crash_path()
    own = s.drop(td[i0 + 10])                                 # seri lain libur di hari terendah target
    m = rg.episode_metrics(own, rg.episode_window(td, td[i0], (-20, 60)), td)
    assert m["return_ke_titik_terendah"] > -0.30 and m["hari_ke_titik_terendah"] != 10


def test_episode_corr_window_vs_pre_and_normal():
    rng = np.random.default_rng(1)
    td = pd.bdate_range("2015-01-01", periods=800)
    i0 = 600
    x = rng.normal(size=len(td))
    t = np.where((np.arange(len(td)) >= i0 - 20) & (np.arange(len(td)) <= i0 + 60), 0.9 * x, 0) + rng.normal(0, 0.4, len(td))
    daily = pd.DataFrame({"Date": td, "id": "X", "r_x": x, "r_target": t})
    w = rg.episode_window(td, td[i0], (-20, 60))
    normal = rg.normal_day_mask(td, [w])
    c = rg.episode_corr(daily, "X", w, td, 250, normal)
    assert c["r_jendela"] > 0.8 and abs(c["r_setahun_sebelum"]) < 0.2 and c["p_fisher"] < 1e-6
    assert c["n_jendela"] == 81 and c["n_setahun_sebelum"] == 250 and abs(c["r_normal"]) < 0.1


def test_event_study_summary_excludes_descriptive():
    td, i0, s = _crash_path()
    close = pd.DataFrame({"T": s, "P": s * 1.0})
    noise = np.random.default_rng(9).normal(0, 0.001, len(td) - 1)
    daily = pd.DataFrame({"Date": td[1:], "id": "P", "r_x": np.diff(np.log(s)) + noise, "r_target": np.diff(np.log(s))})
    es = rg.event_study(close, daily, "T", ["P"], {"a": str(td[i0].date()), "akhir": str(td[380].date())}, (-20, 60), 250)
    assert es.loc[es["episode"] == "akhir", "terpotong"].all() and not es.loc[es["episode"] == "a", "terpotong"].any()
    med = rg.summarize_episodes(es, ["akhir"], ["max_drawdown", "return_kumulatif_akhir"])
    assert med.loc["T", "max_drawdown"] == pytest.approx(-0.30) and med.attrs["episode"] == ["a"]
    paths = rg.event_paths(close, ["T", "P"], rg.episode_window(td, td[i0], (-20, 60)), td)
    assert paths.index.min() == -20 and paths.index.max() == 60 and paths.loc[10, "T"] == pytest.approx(70)


# ---- aset pelindung ---------------------------------------------------------------------------------
def test_crisis_week_mask_and_sign_flip():
    rng = np.random.default_rng(2)
    td = pd.bdate_range("2010-01-01", periods=2000)
    wins = [rg.episode_window(td, td[i], (-20, 60)) for i in (400, 1200)]
    widx = pd.date_range("2010-01-01", td[-1], freq="W-FRI")
    crisis = rg.crisis_week_mask(widx, wins)
    assert 30 <= crisis.sum() <= 40
    x = rng.normal(size=len(widx))
    t = np.where(crisis, -0.7 * x, 0.4 * x) + rng.normal(0, 0.5, len(widx))
    res = rg.crisis_weekly_corr(pd.DataFrame({"T": t, "G": x}, index=widx), "T", ["G"], crisis)
    assert res.loc["G", "tanda_berbalik"] and res.loc["G", "r_krisis"] < 0 < res.loc["G", "r_normal"]


def test_episode_returns_units():
    td, i0, s = _crash_path()
    close = pd.DataFrame({"T": s, "Y": np.linspace(2.0, 1.0, len(td))})
    r = rg.episode_returns(close, "T", ["Y"], {"T": "log", "Y": "diff"}, {"a": str(td[i0].date())}, (-20, 60))
    y = r[r["id"] == "Y"].iloc[0]
    assert y["satuan"] == "pp" and y["ke_titik_terendah_target"] < 0
    assert r[r["id"] == "T"]["ke_titik_terendah_target"].iloc[0] == pytest.approx(-0.30)


# ---- Brent ------------------------------------------------------------------------------------------
def test_quantile_regression_location_scale():
    rng = np.random.default_rng(3)
    n, b, c = 4000, 0.2, -0.3
    x = rng.uniform(0, 2, n)
    y = b * x + 0.5 * (1 + c * x) * rng.normal(size=n)            # skala menyempit saat x naik
    w = pd.DataFrame({"y": y, "x": x}, index=pd.date_range("2000-01-07", periods=n, freq="W-FRI"))
    res = rg.quantile_regression(w, "y", ["x"], [0.1, 0.5, 0.9])
    beta = res.set_index("kuantil")["beta"]
    expect = {q: b + 0.5 * c * stats.norm.ppf(q) for q in (0.1, 0.5, 0.9)}
    for q, e in expect.items():
        assert beta[q] == pytest.approx(e, abs=0.06)
    assert beta[0.1] > beta[0.9] and beta["OLS"] == pytest.approx(b, abs=0.05) and "p_bh" in res


def test_spike_interaction_detects_sign_reversal():
    rng = np.random.default_rng(4)
    n = 3000
    x, z = rng.normal(0, 0.04, n), rng.normal(0, 0.02, n)
    thr = np.quantile(x, 0.9)
    y = np.where(x > thr, -0.4 * x, 0.3 * x) + 0.5 * z + rng.normal(0, 0.01, n)
    w = pd.DataFrame({"y": y, "x": x, "z": z}, index=pd.date_range("1990-01-05", periods=n, freq="W-FRI"))
    res = rg.spike_interaction(w, "y", "x", ["z"], thr)
    assert res.loc["x", "beta"] == pytest.approx(0.3, abs=0.05)
    total = res.loc["x saat lonjakan (total)"]
    assert total["beta"] < 0 and total["ci_hi"] < 0 and res.attrs["n_lonjakan"] == pytest.approx(0.1 * n, abs=2)
