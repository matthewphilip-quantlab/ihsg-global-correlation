import math

import numpy as np
import pandas as pd
import pytest

from src import analysis as an


def _mvn(rho: float, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)


def _weekly_panel(n=520, seed=0):
    """Panel mingguan sintetis: T = 0.5 A - 0.3 B + noise; C hampir kolinear dengan A."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-08", periods=n, freq="W-FRI")
    a, b = rng.normal(0, 0.02, n), rng.normal(0, 0.02, n)
    c = a + rng.normal(0, 0.004, n)
    t = 0.5 * a - 0.3 * b + rng.normal(0, 0.01, n)
    return pd.DataFrame({"T": t, "A": a, "B": b, "C": c}, index=idx)


# ---- dasar ------------------------------------------------------------------------------
def test_fisher_ci_contains_r_and_shrinks_with_n():
    lo, hi = an.fisher_ci(0.4, 100)
    lo2, hi2 = an.fisher_ci(0.4, 1000)
    assert lo < 0.4 < hi and (hi2 - lo2) < (hi - lo)
    slo, shi = an.fisher_ci(0.4, 100, se_factor=an.SPEARMAN_SE_FACTOR)
    assert (shi - slo) > (hi - lo)
    assert all(np.isnan(an.fisher_ci(0.4, 3)))


def test_fisher_diff():
    z, p = an.fisher_diff(0.3, 200, 0.3, 200)
    assert z == 0 and p == pytest.approx(1.0)
    z, p = an.fisher_diff(0.6, 300, 0.1, 300)
    assert z > 0 and p < 0.001


def test_corr_with_ci_recovers_true_rho():
    d = _mvn(0.5, 3000)
    res = an.corr_with_ci(pd.Series(d[:, 0]), pd.Series(d[:, 1]))
    assert res["ci_lo"] < 0.5 < res["ci_hi"] and res["n"] == 3000
    sp = an.corr_with_ci(pd.Series(d[:, 0]), pd.Series(d[:, 1]), "spearman")
    assert 0.4 < sp["r"] < 0.55


def test_in_period_open_end():
    dates = pd.Series(pd.to_datetime(["2009-12-31", "2010-01-04", "2020-06-01"]))
    assert an.in_period(dates, "2010-01-01", None).tolist() == [False, True, True]
    assert an.in_period(dates, "2010-01-01", "2019-12-31").tolist() == [False, True, False]


def test_hac_lags_rule():
    assert an.hac_lags(507) == 5 and an.hac_lags(1000) == 6 and an.hac_lags(100) == 4


# ---- tabel korelasi ------------------------------------------------------------------------
def _daily_pairs(seed=1):
    """'SYNC': target bereaksi di hari yang sama. 'AFTER': target baru bereaksi hari berikutnya."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", "2021-12-31")
    xs, xa = rng.normal(0, 0.01, len(idx)), rng.normal(0, 0.01, len(idx))
    t = 0.6 * xs + 0.6 * np.concatenate([[0], xa[:-1]]) + rng.normal(0, 0.006, len(idx))
    daily = pd.concat([pd.DataFrame({"Date": idx, "id": "SYNC", "r_target": t, "r_x": xs}),
                       pd.DataFrame({"Date": idx, "id": "AFTER", "r_target": t, "r_x": xa})])
    daily["flag_stale_quote"] = False
    prices = pd.DataFrame({"T": np.exp(np.cumsum(t)), "SYNC": np.exp(np.cumsum(xs)), "AFTER": np.exp(np.cumsum(xa))},
                          index=idx)
    weekly = np.log(prices.resample("W-FRI").last()).diff()
    return daily, weekly


def test_correlation_table_and_session_gap():
    daily, weekly = _daily_pairs()
    periods = {"screening": ("2010-01-01", "2019-12-31"), "pasca": ("2020-01-01", None)}
    tab = an.correlation_table(daily, weekly, "T", ["SYNC", "AFTER"], periods)
    assert len(tab) == 2 * 2 * 2 * 2 and (tab["p_bh"] >= tab["p"] - 1e-12).all()
    n_scr = tab.query("periode == 'screening' and frekuensi == 'harian' and id == 'SYNC'")["n"].iloc[0]
    assert n_scr == len(pd.bdate_range("2010-01-01", "2019-12-31"))
    series = pd.DataFrame({"id": ["SYNC", "AFTER"], "name": ["s", "a"], "session": ["sync", "after"]})
    gap = an.session_gap_table(tab, series, "screening")
    assert gap.loc["AFTER", "rasio_harian_mingguan"] < 0.3          # harian meremehkan hubungan
    assert gap.loc["SYNC", "rasio_harian_mingguan"] > 0.7


def test_compare_targets_reports_difference():
    daily, weekly = _daily_pairs()
    periods = {"screening": ("2010-01-01", "2019-12-31")}
    a = an.correlation_table(daily, weekly, "T", ["SYNC"], periods)
    b = a.assign(r=a["r"] + 0.1)
    cmp = an.compare_targets(a, b, "idr", "usd")
    assert np.allclose(cmp["selisih"], 0.1)


# ---- regresi faktor -------------------------------------------------------------------------
def test_factor_regression_recovers_betas_and_vif():
    w = _weekly_panel()
    coef, model = an.factor_regression(w, "T", ["A", "B"], None, None)
    assert coef.loc["A", "ci_lo"] < 0.5 < coef.loc["A", "ci_hi"]
    assert coef.loc["B", "ci_lo"] < -0.3 < coef.loc["B", "ci_hi"]
    assert model["hac_lags"] == an.hac_lags(520) and 0.48 < model["r2"] < 0.68   # teori: 1.36 / 2.36 = 0.58
    vif = an.vif_table(w, ["A", "B", "C"], None, None)
    assert vif["A"] > 5 and vif["C"] > 5 and vif["B"] < 2
    one = an.single_factor_betas(w, "T", ["A", "C"], None, None)
    assert 0.4 < one.loc["C", "beta_1f"] < 0.6                       # satu faktor: C menyerap efek A


def test_scenario_text_uses_config_phrase_and_shock():
    scfg = {"default_shock_pct": 1, "by_id": {"VIX": {"shock_pct": 10, "phrase": "VIX naik 10%"}}}
    txt = an.scenario_text("VIX", "VIX", -0.05, -0.07, -0.03, scfg)
    assert txt.startswith("VIX naik 10%") and "-0.50%" in txt and "dikendalikan" in txt
    assert an.scenario_text("STI", "Straits Times", 0.8, 0.6, 1.0, scfg).startswith("Straits Times naik 1%")


# ---- korelasi asimetris ------------------------------------------------------------------------
def test_expected_conditional_corr_matches_bgl_formula():
    rho = 0.6
    data = _mvn(rho, 200_000, seed=3)
    e_down, e_up = an.expected_conditional_corr(data, "tanda", 0.1, 400_000, seed=4)
    ratio = 1 / (1 - 2 / math.pi)                    # Var(X) / Var(X | X > 0) untuk normal standar
    bgl = rho / math.sqrt(rho ** 2 + (1 - rho ** 2) * ratio)
    assert e_down == pytest.approx(bgl, abs=0.01) and e_up == pytest.approx(bgl, abs=0.01)
    assert bgl < rho                                  # memotong sampel menurunkan korelasi


def test_stationary_bootstrap_indices():
    rng = np.random.default_rng(0)
    idx = an.stationary_bootstrap_indices(500, 200, 4, rng)
    assert idx.shape == (200, 500) and idx.min() >= 0 and idx.max() < 500
    cont = (np.diff(idx, axis=1) % 500 == 1).mean()
    assert 0.7 < cont < 0.8                           # P(lanjut blok) = 1 - 1/4


ACFG = {"exceedance_q": 0.1, "n_boot": 200, "mean_block_weeks": 4, "n_mc": 50_000, "seed": 1}


def test_asymmetry_absent_under_normality():
    d = _mvn(0.5, 1000, seed=5)
    w = pd.DataFrame({"T": d[:, 0], "X": d[:, 1]}, index=pd.date_range("2000-01-07", periods=1000, freq="W-FRI"))
    res = an.asymmetric_table(w, "T", ["X"], [], None, None, ACFG, modes=("tanda",))
    r = res.iloc[0]
    assert r["kondisi"] == "sendiri" and r["r_turun"] < 0.5 and r["r_naik"] < 0.5   # efek pemotongan
    assert abs(r["asimetri_berlebih"]) < 0.1 and r["ci_lo_boot"] < 0 < r["ci_hi_boot"]


def test_asymmetry_detected_when_crashes_are_shared():
    rng = np.random.default_rng(6)
    n = 1500
    x = rng.normal(0, 1, n)
    t = np.where(x < 0, 0.8 * x, 0.1 * x) + rng.normal(0, 0.4, n)    # ikut jatuh, tidak ikut naik
    w = pd.DataFrame({"T": t, "X": x, "G": x}, index=pd.date_range("2000-01-07", periods=n, freq="W-FRI"))
    res = an.asymmetric_table(w, "T", ["X"], ["G"], None, None, ACFG)
    row = res[(res["kondisi"] == "G") & (res["mode"] == "tanda")].iloc[0]
    assert row["r_turun"] > row["r_naik"] and row["ci_lo_boot"] > 0 and row["p_fisher"] < 0.001
    assert set(res["mode"]) == {"tanda", "exceedance"} and {"p_boot_bh", "p_fisher_bh"} <= set(res.columns)
    exc = res[(res["kondisi"] == "G") & (res["mode"] == "exceedance")].iloc[0]
    assert exc["n_turun"] == pytest.approx(0.1 * n, abs=2)


# ---- matriks, sub-periode, sensitivitas -------------------------------------------------------
def test_corr_matrix_symmetric():
    m = an.corr_matrix(_weekly_panel(), ["T", "A", "B"], None, None)
    assert np.allclose(m, m.T) and np.allclose(np.diag(m), 1)


def test_subperiod_test_detects_change():
    a, b = _mvn(0.05, 260, seed=7), _mvn(0.6, 260, seed=8)
    idx = pd.date_range("2010-01-08", periods=520, freq="W-FRI")
    w = pd.DataFrame(np.vstack([a, b]), columns=["T", "X"], index=idx)
    split = [[str(idx[0].date()), str(idx[259].date())], [str(idx[260].date()), str(idx[-1].date())]]
    res = an.subperiod_test({"bersih": w}, "T", "X", split)
    assert res.loc[0, "selisih"] > 0.3 and res.loc[0, "p_fisher"] < 0.001


def test_stale_sensitivity_drops_flagged():
    daily, _ = _daily_pairs()
    daily = daily[daily["id"] == "SYNC"].copy()
    daily.loc[daily.index[:30], "flag_stale_quote"] = True
    res = an.stale_sensitivity(daily, "SYNC", {"penuh": ("2010-01-01", None)})
    n_all = res.query("sampel == 'semua return' and metode == 'pearson'")["n"].iloc[0]
    n_clean = res.query("sampel == 'tanpa flag_stale_quote' and metode == 'pearson'")["n"].iloc[0]
    assert n_all - n_clean == 30 and (res["n_flag"] == 30).all()


def test_rank_stability_and_outputs(tmp_path):
    daily, weekly = _daily_pairs()
    periods = {"screening": ("2010-01-01", "2019-12-31"), "pasca": ("2020-01-01", None), "penuh": ("2010-01-01", None)}
    tab = an.correlation_table(daily, weekly, "T", ["SYNC", "AFTER"], periods)
    ranks = an.rank_stability(tab)
    assert set(ranks.columns) == set(periods) and ranks.loc["SYNC", "screening"] in (1.0, 2.0)
    paths = an.save_phase3_outputs(tmp_path, {"korelasi": tab, "rank": ranks}, {"catatan": "uji"})
    assert all(p.exists() for p in paths) and (tmp_path / "summary.json").exists()


def test_scenario_lines_warn_on_sign_flip_and_vif():
    coef = pd.DataFrame({"beta": [-0.16, 0.30], "p_bh": [0.03, 0.001], "vif": [4.2, 6.0], "beta_1f": [0.37, 0.4],
                         "ci_lo_1f": [0.25, 0.3], "ci_hi_1f": [0.48, 0.5], "r2_1f": [0.11, 0.3], "label": ["", "lbl"]},
                        index=["SPX", "EEM"])
    lines = an.scenario_lines(coef, {"SPX": "S&P naik 1%", "EEM": "EEM naik 1%"}, 5.0)
    assert "Tanda berbalik" in lines[0] and "VIF 4.2" in lines[0]
    assert "Tanda berbalik" not in lines[1] and "VIF 6.0" in lines[1] and "*(lbl)*" in lines[1]


def test_usd_effect_text_magnitude_aware():
    cmp = pd.DataFrame({"frekuensi": "mingguan", "metode": "pearson", "periode": ["a", "a", "b", "b", "b"],
                        "id": ["X", "Y", "X", "Y", "FX"], "r_idr": [0.50, -0.30, 0.40, -0.30, -0.3],
                        "r_usd": [0.51, -0.30, 0.46, -0.36, -0.6]})
    txt = an.usd_effect_text(cmp, "FX")
    assert "a +0.01 (praktis sama)" in txt and "b +0.06 (lebih kuat dalam USD)" in txt   # VIX-like -0.30 -> -0.36 = +0.06
