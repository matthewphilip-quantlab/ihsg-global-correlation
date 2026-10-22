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


def test_log_append_only_fill_and_evaluate(tmp_path):
    cfg, close = _panel(end="2021-03-30")
    m = fw.freeze_models(cfg, close, "2020-12-31")
    path = tmp_path / "log.csv"
    sig = fw.next_signal(cfg, close, m)
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
    ev = fw.evaluate_log(path)
    assert ev["n"].iloc[0] == 1                                                      # baris terlambat dibuang


def test_signal_window_blocks_live_us_bar_and_late_logging():
    fd = pd.Timestamp("2026-10-22")                                       # Kamis
    assert not fw.signal_window(fd, pd.Timestamp("2026-10-22 14:06", tz="UTC"))[0]    # 21:06 WIB: sesi AS berjalan
    ok, _ = fw.signal_window(fd, pd.Timestamp("2026-10-22 22:00", tz="UTC"))          # 05:00 WIB 23 Okt
    assert ok
    late, why = fw.signal_window(fd, pd.Timestamp("2026-10-23 02:30", tz="UTC"))      # 09:30 WIB: BEI sudah buka
    assert not late and "terlambat" in why
    assert fw.signal_window(pd.Timestamp("2026-10-23"), pd.Timestamp("2026-10-25 23:00", tz="UTC"))[0]   # Jumat -> Senin
