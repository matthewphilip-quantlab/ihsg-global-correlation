import numpy as np
import pandas as pd
import pytest

from src import volatility as vol


def _simulate_garchx(n=4000, mu=0.05, omega=0.03, alpha=0.08, beta=0.88, gamma=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.06, n)
    s2 = np.empty(n)
    e = np.empty(n)
    s2[0] = omega / (1 - alpha - beta)
    e[0] = np.sqrt(s2[0]) * rng.normal()
    for t in range(1, n):
        s2[t] = (omega + alpha * e[t - 1] ** 2 + beta * s2[t - 1]) * np.exp(gamma * x[t - 1])
        e[t] = np.sqrt(s2[t]) * rng.normal()
    return mu + e, x


def test_arch_backcast_formula():
    r = np.arange(1, 101, dtype=float)
    w = 0.94 ** np.arange(75)
    assert vol.arch_backcast(r) == pytest.approx(np.sum(r[:75] ** 2 * w) / w.sum())


def test_matches_arch_when_gamma_zero():
    r, _ = _simulate_garchx()
    v = vol.validate_against_arch(r)
    assert v.loc["loglik sendiri di parameter arch", "selisih_abs"] < 1e-6       # rekursi & backcast identik
    assert v.loc["loglik (estimasi masing-masing)", "selisih_abs"] < 1e-3
    assert (v.loc[["mu", "omega", "alpha", "beta"], "selisih_relatif"] < 1e-2).all()


def test_fit_garchx_recovers_gamma():
    r, x = _simulate_garchx(n=6000, gamma=4.0, seed=1)
    fit = vol.fit_garchx(r, x)
    assert fit["converged"] and fit["params"][4] == pytest.approx(4.0, abs=1.0)
    null = vol.fit_garchx(r, x, gamma_fixed=0.0)
    assert fit["loglik"] > null["loglik"] + 10 and null["params"][4] == 0.0


def test_garch_walk_forward_uses_only_past():
    r, x = _simulate_garchx(n=1300, gamma=2.0, seed=2)
    idx = pd.bdate_range("2014-01-01", periods=len(r))
    rs, xs = pd.Series(r, index=idx), pd.Series(x, index=idx)
    pred = vol.garch_walk_forward(rs, xs, "2014-01-01", "2018-06-01", use_x=True)
    day = pred.index[pred.index.month == pred.index[0].month][5]
    r2 = rs.copy()
    r2.loc[day:] = r2.loc[day:] * 3                 # ubah return hari itu dan sesudahnya
    pred2 = vol.garch_walk_forward(r2, xs, "2014-01-01", "2018-06-01", use_x=True)
    assert pred.loc[day, "s2"] == pytest.approx(pred2.loc[day, "s2"])          # s2_t hanya pakai info s/d t-1
    assert pred2.loc[pred2.index > day, "s2"].iloc[0] != pytest.approx(pred.loc[pred.index > day, "s2"].iloc[0])


def test_qlike_minimised_at_true_variance():
    rng = np.random.default_rng(3)
    proxy = rng.normal(0, np.sqrt(2.0), 200_000) ** 2           # E[proxy] = 2
    grid = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    losses = [vol.qlike(proxy, np.full_like(proxy, g)).mean() for g in grid]
    assert grid[int(np.argmin(losses))] == 2.0
    assert vol.mse(np.array([1.0]), np.array([3.0]))[0] == 4.0


def test_weekly_rv_and_har_frame():
    idx = pd.bdate_range("2020-01-06", periods=30)
    r = pd.Series(np.arange(1, 31, dtype=float) / 10, index=idx)
    rv = vol.weekly_rv(r)
    assert rv["rv"].iloc[0] == pytest.approx(sum((np.arange(1, 6) / 10) ** 2)) and rv["n_hari"].iloc[0] == 5
    vix = pd.Series(np.linspace(15, 25, 30), index=idx)
    f = vol.har_frame(rv, vix, [1, 4])
    assert f["log_rv_4"].iloc[:3].isna().all() and np.isclose(f["log_rv_4"].iloc[3], np.log(rv["rv"].iloc[:4].mean()))
    assert np.isclose(f["log_rv_next"].iloc[0], np.log(rv["rv"].iloc[1])) and f["target_week"].iloc[0] == rv.index[1]
    assert np.isclose(f["dlog_vix"].iloc[1], np.log(vix.iloc[9] / vix.iloc[4]))


def test_har_walk_forward_and_compare_losses():
    r, x = _simulate_garchx(n=2600, gamma=3.0, seed=4)
    idx = pd.bdate_range("2010-01-01", periods=len(r))
    rs = pd.Series(r, index=idx)
    vix = pd.Series(20 * np.exp(np.cumsum(x)), index=idx)
    f = vol.har_frame(vol.weekly_rv(rs), vix, [1, 4, 13])
    base = vol.har_walk_forward(f, ["log_rv_1", "log_rv_4", "log_rv_13"], "2010-01-01", "2017-01-01")
    withx = vol.har_walk_forward(f, ["log_rv_1", "log_rv_4", "log_rv_13", "dlog_vix"], "2010-01-01", "2017-01-01")
    assert (base["s2"] > 0).all() and (base["target_week"] >= pd.Timestamp("2017-01-01")).all()
    cmp = vol.compare_losses(withx, base, "rv")
    assert set(cmp.index) == {"QLIKE", "MSE"} and {"dengan_vix", "tanpa_vix", "dm_hln", "p"} <= set(cmp.columns)
