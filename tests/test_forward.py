import numpy as np
import pandas as pd
import pytest

from src import forward as fw
from src import oos
from src import volatility as vol
from src.config import load_config


def _panel(end="2021-03-31", seed=0):
    """Panel close sintetis berisi seri yang dipakai model utama."""
    cfg = load_config()
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2009-06-01", end)
    ids = {cfg["project"]["target_id"]: 5000, "EEM": 40, "SPX": 2000, "VIX": 20, cfg["oos"]["eido_id"]: 25,
           cfg["usdidr_retest"]["id"]: 13000}
    close = pd.DataFrame({k: v * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))) for k, v in ids.items()}, index=idx)
    return cfg, close


def test_freeze_matches_direct_fit_and_save_once(tmp_path):
    cfg, close = _panel()
    m = fw.freeze_models(cfg, close, "2020-12-31")
    frame, feats, ret, x = fw.build_inputs(cfg, close.loc[:"2020-12-31"])
    train = frame.loc[cfg["oos"]["train_start"]:]
    train = train[train["target_date"] <= pd.Timestamp("2020-12-31")].dropna(subset=feats + ["y"])
    direct = oos._fit_logit(train[feats], train["y"])
    assert np.allclose(pd.Series(m["logit_params"])[direct.params.index], direct.params)
    assert m["model_version"] == "frozen_2020-12-31" and m["n_train_logit"] == len(train)
    p = fw.save_frozen(m, tmp_path)
    assert fw.load_frozen(p)["garch_params"] == m["garch_params"]
    with pytest.raises(FileExistsError):
        fw.save_frozen(m, tmp_path)


def test_next_signal_uses_frozen_params_and_garch_recursion():
    cfg, close = _panel()
    m = fw.freeze_models(cfg, close, "2020-12-31")
    sig = fw.next_signal(cfg, close, m)
    assert sig["feature_date"] == close.index.max() and 0 < sig["p_naik"] < 1 and sig["sigma_pct"] > 0
    _, _, ret, x = fw.build_inputs(cfg, close)
    gp = m["garch_params"]
    pvec = np.array([gp[k] for k in ("mu", "omega", "alpha", "beta", "gamma")])
    r, xx = ret.loc[cfg["oos"]["train_start"]:].to_numpy(), x.loc[cfg["oos"]["train_start"]:].to_numpy()
    s2, _ = vol.garchx_sigma2(pvec, np.r_[r, 0.0], np.r_[xx, 0.0])     # satu langkah ke depan
    assert sig["sigma_pct"] == pytest.approx(np.sqrt(s2[-1]))


def test_next_signal_refuses_when_us_session_missing():
    cfg, close = _panel()
    m = fw.freeze_models(cfg, close, "2020-12-31")
    close.loc[close.index.max(), "EEM"] = np.nan                  # sesi AS hari terakhir belum ada
    with pytest.raises(ValueError, match="belum lengkap"):
        fw.next_signal(cfg, close, m)


def test_freeze_comparators_same_training_data():
    cfg, close = _panel()
    m = fw.freeze_models(cfg, close, "2020-12-31")
    c = fw.freeze_comparators(cfg, close, "2020-12-31", m)
    frame, feats, _, _ = fw.build_inputs(cfg, close.loc[:"2020-12-31"])
    train = frame.loc[cfg["oos"]["train_start"]:]
    train = train[train["target_date"] <= pd.Timestamp("2020-12-31")].dropna(subset=feats + ["y"])
    assert c["baseline_p_naik"] == pytest.approx(train["y"].mean()) and c["garch_tanpa_vix_params"]["gamma"] == 0.0
    assert c["file"] == "frozen_2020-12-31_pembanding"
    with pytest.raises(ValueError, match="tidak identik"):
        fw.freeze_comparators(cfg, close, "2020-11-30", m)
    sig = fw.next_signal(cfg, close, m, c)
    assert sig["p_naik_baseline"] == c["baseline_p_naik"] and sig["sigma_pct_tanpa_vix"] > 0


def test_log_append_only_fill_and_evaluate(tmp_path):
    cfg, close = _panel(end="2021-03-30")
    m = fw.freeze_models(cfg, close, "2020-12-31")
    path = tmp_path / "log.csv"
    sig = fw.next_signal(cfg, close, m, fw.freeze_comparators(cfg, close, "2020-12-31", m))
    on_time = (sig["feature_date"] + pd.Timedelta(hours=20)).tz_localize("UTC").isoformat()
    assert fw.append_log(sig, path, close.index.max(), now_utc=on_time)
    assert not fw.append_log(sig, path, close.index.max(), now_utc=on_time)          # tidak dobel, tidak ditimpa
    late = {**sig, "feature_date": sig["feature_date"] - pd.offsets.BDay(1)}
    fw.append_log(late, path, close.index.max(), now_utc=on_time)                    # tercatat setelah hari target
    _, close2 = _panel(end="2021-03-31")
    assert fw.fill_outcomes(path, close2[cfg["project"]["target_id"]]) == 2
    log = fw.read_log(path)
    assert log["p_naik"].iloc[0] == pytest.approx(round(sig["p_naik"], 6)) and log["y_aktual"].notna().all()
    assert list(fw.on_time(log)) == [True, False]
    ev = fw.evaluate_log(path)                                                       # pra-registrasi: < 250 baris
    assert ev["n_valid"].iloc[0] == 1 and "belum dievaluasi" in ev["status"].iloc[0] and "dm_arah_p" not in ev
    ev1 = fw.evaluate_log(path, min_rows=1)                                          # hanya untuk menguji perhitungan
    assert ev1["status"].iloc[0] == "dievaluasi" and {"dm_arah_p", "dm_qlike_p", "log_loss_baseline"} <= set(ev1.columns)


def test_signal_window_blocks_live_us_bar_and_late_logging():
    fd = pd.Timestamp("2026-10-22")                                       # Kamis
    assert not fw.signal_window(fd, pd.Timestamp("2026-10-22 14:06", tz="UTC"))[0]    # 21:06 WIB: sesi AS berjalan
    ok, _ = fw.signal_window(fd, pd.Timestamp("2026-10-22 22:00", tz="UTC"))          # 05:00 WIB 23 Okt
    assert ok
    late, why = fw.signal_window(fd, pd.Timestamp("2026-10-23 02:30", tz="UTC"))      # 09:30 WIB: BEI sudah buka
    assert not late and "terlambat" in why
    assert fw.signal_window(pd.Timestamp("2026-10-23"), pd.Timestamp("2026-10-25 23:00", tz="UTC"))[0]   # Jumat -> Senin


# ---- jadwal & hari libur ---------------------------------------------------------------------------------
def test_schedule_0915_melbourne_inside_window_all_year_both_dst():
    chk = fw.schedule_check(9, 15, "Australia/Melbourne", "2026-10-26", "2029-10-26")
    assert chk["ok"].all() and len(chk) > 700
    assert chk["menit_setelah_batas_awal"].min() >= 44 and chk["menit_sebelum_bei_buka"].min() >= 164
    mon = chk[pd.to_datetime(chk["jadwal_lokal"]).dt.dayofweek == 0]
    assert (pd.to_datetime(mon["feature_date"]).dt.dayofweek == 4).all()          # Senin pagi Melbourne -> fitur Jumat
    early = fw.schedule_check(8, 0, "Australia/Melbourne", "2027-01-04", "2027-01-08")                 # AEDT: 21:00 UTC
    assert early.iloc[0]["ok"] and not early.iloc[1:]["ok"].any()     # Senin sah (fitur Jumat); Selasa-Jumat terlalu awal
    assert not fw.schedule_check(13, 0, "Australia/Melbourne", "2027-07-05", "2027-07-09")["ok"].any()  # AEST: BEI sudah buka


def test_friday_features_predict_next_bei_day_including_monday_holiday(tmp_path):
    cfg, close = _panel(end="2021-03-26")                                       # Jumat
    m = fw.freeze_models(cfg, close, "2020-12-31")
    sig = fw.next_signal(cfg, close, m)
    assert sig["feature_date"].dayofweek == 4
    assert fw.signal_window(sig["feature_date"], pd.Timestamp("2021-03-28 22:15", tz="UTC"))[0]   # Senin 09:15 Melbourne
    path = tmp_path / "log.csv"
    fw.append_log(sig, path, close.index.max(), now_utc="2021-03-28T22:15:00+00:00")
    _, later = _panel(end="2021-03-31")
    later_t = later[cfg["project"]["target_id"]].drop(pd.Timestamp("2021-03-29"))   # Senin libur BEI
    fw.fill_outcomes(path, later_t)
    assert pd.Timestamp(fw.read_log(path)["target_date"].iloc[0]) == pd.Timestamp("2021-03-30")   # target = Selasa


def test_bei_holiday_midweek_is_skipped_not_mislogged():
    # BEI libur Selasa: run Rabu pagi Melbourne (Selasa 22:15 UTC) hanya melihat IHSG s/d Senin -> terlambat, tidak dicatat
    ok, why = fw.signal_window(pd.Timestamp("2027-03-08"), pd.Timestamp("2027-03-09 22:15", tz="UTC"))
    assert not ok and "terlambat" in why
    # Mac bangun jam 10:00 Melbourne (setelah BEI buka): launchd menjalankan job, script menolak
    assert not fw.signal_window(pd.Timestamp("2027-03-09"), pd.Timestamp("2027-03-09 23:00", tz="UTC") + pd.Timedelta(hours=3))[0]
