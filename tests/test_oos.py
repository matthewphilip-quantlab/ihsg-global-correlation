import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import oos


def _predictive_frame(n=3000, beta=1.2, seed=0, start="2010-01-01"):
    """Frame sintetis: y(t+1) ~ Bernoulli(logistic(beta * f1(t))), f2 noise murni."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    p = 1 / (1 + np.exp(-beta * f1))
    y = (rng.random(n) < p).astype(float)
    return pd.DataFrame({"f1": f1, "f2": f2, "y": y, "target_date": pd.Series(idx, index=idx).shift(-1).to_numpy()},
                        index=idx).iloc[:-1]


# ---- fitur & target ----------------------------------------------------------------------------
def test_next_day_target_alignment():
    idx = pd.bdate_range("2020-01-01", periods=5)
    s = pd.Series([100, 101, 100, 100, 103.0], index=idx)
    t = oos.next_day_target(s)
    assert list(t.index) == list(idx[1:])
    assert t.loc[idx[1], "target_date"] == idx[2] and t.loc[idx[1], "y"] == 0          # 101 -> 100
    assert t.loc[idx[3], "y"] == 1 and np.isnan(t.loc[idx[4], "y"])                     # tidak ada t+1
    assert t.loc[idx[2], "y"] == 0                                                      # return nol = tidak naik


def test_feature_frame_same_date_without_ffill_and_premium():
    idx = pd.bdate_range("2020-01-01", periods=6)
    close = pd.DataFrame({"T": [5000, 5050, 5000, 5100, 5150, 5200.0],
                          "A": [10, 11, np.nan, 12, 12.5, 13.0],
                          "FX": [14000, 14100, 14050, 14000, 13950, 13900.0],
                          "E": [25, 25.5, 25.2, 25.8, 26.1, 26.0]}, index=idx)
    f = oos.feature_frame(close, "T", ["A"], {}, fx_id="FX", eido_id="E")
    assert np.isnan(f.loc[idx[2], "r_A"])                                   # A tidak diperdagangkan -> tidak diisi
    assert np.isclose(f.loc[idx[3], "r_A"], np.log(12 / 11))                # return lintas libur di kalender A sendiri
    prem = np.log(close["E"]) - np.log(close["T"] / close["FX"])
    assert np.isclose(f.loc[idx[3], "d_premium"], prem[idx[3]] - prem[idx[2]])
    audit = oos.timing_audit(f, ["r_A", "d_premium"])
    assert audit["target_setelah_fitur"] and audit["selisih_hari_min"] >= 1


# ---- metrik ------------------------------------------------------------------------------------------
def test_metrics_and_auc():
    y = np.array([0, 0, 1, 1.0])
    assert oos.auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert oos.auc(y, np.array([0.5] * 4)) == 0.5
    rng = np.random.default_rng(1)
    yy, pp = rng.integers(0, 2, 500), rng.random(500)
    mw = stats.mannwhitneyu(pp[yy == 1], pp[yy == 0]).statistic / ((yy == 1).sum() * (yy == 0).sum())
    assert oos.auc(yy, pp) == pytest.approx(mw)
    m = oos.metrics(y, np.array([0.4, 0.6, 0.6, 0.9]))
    assert m["hit_rate"] == 0.75 and m["brier"] == pytest.approx((0.16 + 0.36 + 0.16 + 0.01) / 4)
    assert m["log_loss"] == pytest.approx(-np.mean(np.log([0.6, 0.4, 0.6, 0.9])))


def test_calibration_table_bins():
    rng = np.random.default_rng(2)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(float)
    cal = oos.calibration_table(y, p, 10)
    assert len(cal) == 10 and cal["n"].sum() == 1000
    assert np.corrcoef(cal["p_rata"], cal["frekuensi_naik"])[0, 1] > 0.9


def test_diebold_mariano_direction():
    rng = np.random.default_rng(3)
    base = rng.exponential(1, 2000)
    better = base - 0.05 + rng.normal(0, 0.1, 2000)
    dm = oos.diebold_mariano(better, base)
    assert dm["selisih_loss_rata"] < 0 and dm["p"] < 1e-6 and dm["p_satu_sisi_model_lebih_baik"] < 1e-6
    same = oos.diebold_mariano(base + rng.normal(0, 0.1, 2000), base)
    assert same["p"] > 0.01


# ---- walk-forward ----------------------------------------------------------------------------------
def test_walk_forward_beats_baseline_when_signal_exists():
    f = _predictive_frame()
    pred = oos.walk_forward(f, {"m": ["f1"], "noise": ["f2"]}, "2010-01-01", "2018-01-01")
    met = oos.metrics_table(oos.common_rows(pred, ["m", "noise"]), ["baseline", "m", "noise"]).loc["OOS penuh"]
    assert met.loc["m", "log_loss"] < met.loc["baseline", "log_loss"] - 0.05
    assert abs(met.loc["noise", "log_loss"] - met.loc["baseline", "log_loss"]) < 0.01
    assert pred.index.min() >= pd.Timestamp("2018-01-01") and pred["refit"].nunique() > 6


def test_walk_forward_never_trains_on_unrealised_targets():
    """Mengubah y untuk baris yang targetnya terealisasi di/sesudah awal bulan tidak boleh mengubah prediksi bulan itu."""
    f = _predictive_frame(n=1500)
    m0 = pd.Timestamp("2015-06-01")
    base = oos.walk_forward(f, {"m": ["f1"]}, "2010-01-01", "2015-06-01")
    g = f.copy()
    g.loc[g["target_date"] >= m0, "y"] = 1 - g.loc[g["target_date"] >= m0, "y"]
    alt = oos.walk_forward(g, {"m": ["f1"]}, "2010-01-01", "2015-06-01")
    june = base.index < pd.Timestamp("2015-07-01")
    assert np.allclose(base.loc[june, "p_m"], alt.loc[june, "p_m"])
    assert np.allclose(base.loc[june, "p_baseline"], alt.loc[june, "p_baseline"])


def test_trend_spread_z_uses_train_parameters_only():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2010-01-01", periods=2000)
    x = np.cumsum(rng.normal(0, 0.01, 2000))
    t = np.arange(2000) / 252
    e = np.zeros(2000)
    for i in range(1, 2000):
        e[i] = 0.9 * e[i - 1] + rng.normal(0, 0.005)
    df = pd.DataFrame({"log_target_usd": x, "log_eido": 0.1 + 0.03 * t + x + e}, index=idx)
    tr, te = df.iloc[:1500], df.iloc[1500:]
    ztr, zte = oos.trend_spread_z(tr, te, idx[0])
    assert abs(ztr.mean()) < 1e-8 and ztr.std() == pytest.approx(1, rel=1e-6)
    _, zte2 = oos.trend_spread_z(tr, te.assign(log_eido=te["log_eido"] + 1), idx[0])
    shift = (zte2 - zte).to_numpy()
    assert np.ptp(shift) < 1e-9 and shift[0] > 0          # parameter tetap dari data latih: geser konstan 1/sd_latih


# ---- placebo ------------------------------------------------------------------------------------
def test_shuffled_target_placebo_not_better_than_baseline_on_average():
    """Target acak -> koefisien bertanda acak. Rata-rata lintas pengacakan tidak boleh lebih baik dari baseline,
    sedangkan model asli jelas lebih baik."""
    f = _predictive_frame(n=2500, beta=0.6)
    res = oos.shuffled_target_placebo(f, ["f1"], "2010-01-01", "2017-01-01", seeds=8)
    s = oos.placebo_summary(res)
    assert s["rata_selisih_log_loss"] > -0.002 and s["n_pengacakan"] == 8
    real = oos.common_rows(oos.walk_forward(f, {"m": ["f1"]}, "2010-01-01", "2017-01-01"), ["m"])
    real_gain = oos.metrics(real["y"], real["p_m"])["log_loss"] - oos.metrics(real["y"], real["p_baseline"])["log_loss"]
    assert real_gain < res["selisih_log_loss"].min() and real_gain < -0.02


def test_leaked_features_shift_and_jump():
    f = _predictive_frame(n=2500)
    f["leaky"] = f["y"].shift(1)            # nilai di baris t+1 = y(t)
    leaked = oos.leaked_features(f, ["leaky"])
    assert np.allclose(leaked["leaky"].iloc[:-1].to_numpy(), f["y"].iloc[:-1].to_numpy())
    pred = oos.walk_forward(leaked, {"bocor": ["leaky"]}, "2010-01-01", "2017-01-01")
    met = oos.metrics_table(oos.common_rows(pred, ["bocor"]), ["baseline", "bocor"]).loc["OOS penuh"]
    assert met.loc["bocor", "hit_rate"] > 0.95 and "konvergen_bocor" in pred


def test_phase6_summary_text():
    mm = pd.DataFrame({"n": [500, 500], "hit_rate": [0.52, 0.56], "log_loss": [0.692, 0.680], "auc": [0.5, 0.6]},
                      index=["baseline", "utama"])
    leak = oos.leak_verdict(pd.DataFrame({"hit_rate": [0.99, 0.55], "auc": [0.999, 0.59]}, index=["bocor", "halus"]),
                            0.56, 0.60)
    vol_tab = pd.DataFrame({"dengan_vix": [1.0], "tanpa_vix": [1.1], "selisih_loss_rata": [-0.1], "p": [0.01]},
                           index=["QLIKE"])
    txt = oos.phase6_summary(mm, {"p": 0.001, "selisih_loss_rata": -0.012},
                             {"n_pengacakan": 10, "rata_selisih_log_loss": 0.001, "rata_auc": 0.5}, leak,
                             {"GARCH-X": vol_tab}, {"hit_rate": 0.51, "hit_rate_naif": 0.5, "p": 0.4})
    assert "signifikan" in txt and "VIX **membantu**" in txt and "Tradeability" in txt and "EKSPLORATIF" in txt
    assert "bocor: hit rate 99.0%, AUC 0.999 (melonjak)" in txt and "halus: hit rate 55.0%, AUC 0.590 (TIDAK melonjak)" in txt


def test_hit_rate_test_counts_disagreements():
    y = np.array([1, 1, 0, 0, 1, 0.0])
    res = oos.hit_rate_test(y, np.array([0.6, 0.6, 0.4, 0.4, 0.6, 0.6]), np.full(6, 0.55))
    assert res["hanya_model_benar"] == 2 and res["hanya_baseline_benar"] == 0 and res["p"] == 0.5
