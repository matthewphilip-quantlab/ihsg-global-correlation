import numpy as np
import pandas as pd

from src import quality as q
from tests.conftest import make_ohlc, random_walk


# ---- gerakan besar & bad tick ----------------------------------------------------
def test_reverting_spike_is_bad_tick_but_persistent_jump_is_not():
    s = random_walk(sd=0.002)
    spike_day, jump_day = s.index[300], s.index[600]
    s.loc[spike_day] *= 1.20                      # salah harga satu hari
    s.loc[jump_day:] *= 1.20                      # loncatan permanen (gerakan nyata)
    mv = q.flag_moves(s, "log", threshold=0.05)
    assert bool(mv.loc[spike_day, "bad_tick"])
    assert not bool(mv.loc[s.index[301], "bad_tick"])   # hari pembalikan: besar, tapi bukan bad tick
    assert not bool(mv.loc[jump_day, "bad_tick"])


def test_vix_threshold_avoids_false_positive(qcfg):
    s = random_walk(sd=0.002, level=15)
    s.iloc[100:] *= 1.25                          # VIX naik 25% dalam sehari: wajar
    vix = q.flag_moves(s, "log", qcfg["spike_threshold"]["sentiment"])
    idx = q.flag_moves(s, "log", qcfg["spike_threshold"]["index"])
    assert vix.empty and len(idx) == 1


def test_series_quality_flags_unavailable_and_late(qcfg):
    meta = dict(id="X", ticker="X", name="X", group="core", category="index", session="sync", transform="log")
    assert q.series_quality(None, meta, pd.Timestamp("2007-01-01"), qcfg)["status"] == "tidak tersedia"
    late = make_ohlc(random_walk(start="2010-05-07"))
    res = q.series_quality(late, meta, pd.Timestamp("2007-01-01"), qcfg)
    assert res["status"].startswith("mulai 2010-05-07")


# ---- keputusan periode -------------------------------------------------------------
def test_decide_start_uses_2007_when_complete():
    d = q.decide_start(random_walk(), "2007-01-01", "2010-01-01", min_obs_per_year=200)
    assert d["start_used"] == pd.Timestamp("2007-01-01") and not d["used_fallback"]


def test_decide_start_falls_back_when_late():
    d = q.decide_start(random_walk(start="2008-06-02"), "2007-01-01", "2010-01-01", 200)
    assert d["start_used"] == pd.Timestamp("2010-01-01") and d["used_fallback"]


def test_decide_start_falls_back_when_early_year_thin():
    s = random_walk()
    s = s.drop(s.loc["2008-03-01":"2008-09-30"].index)    # 2008 bolong
    d = q.decide_start(s, "2007-01-01", "2010-01-01", 200)
    assert d["used_fallback"] and "2008" in d["reason"]


# ---- sanity check 2008 ------------------------------------------------------------
def _jkse_2008_like(ref):
    s = random_walk(start="2007-01-01", end="2009-12-31", sd=0.005, level=2000)
    s = s.drop([pd.Timestamp(x) for x in ref["suspended_days"]])
    s.loc[pd.Timestamp(ref["peak"]["date"])] = ref["peak"]["close"]
    s.loc[pd.Timestamp(ref["trough"]["date"])] = ref["trough"]["close"]
    for drop in ref["daily_drops"]:
        d = pd.Timestamp(drop["date"])
        prev = s.loc[:d].iloc[-2]
        s.loc[d] = prev * (1 + drop["ret_pct"] / 100)
    return s


def test_sanity_check_passes_on_matching_data(cfg):
    ref = cfg["sanity_checks"]["jkse_2008"]
    res = q.sanity_check_2008(_jkse_2008_like(ref), ref)
    assert (res["status"] == "PASS").all(), res


def test_sanity_check_warns_on_stale_suspended_row(cfg):
    ref = cfg["sanity_checks"]["jkse_2008"]
    s = _jkse_2008_like(ref)
    d = pd.Timestamp(ref["suspended_days"][0])
    s.loc[d] = s.loc[:d].iloc[-1]                  # Yahoo mengisi hari suspend dengan harga kemarin
    s = s.sort_index()
    res = q.sanity_check_2008(s, ref)
    assert res.loc[res["tanggal"] == d.date(), "status"].item() == "WARN"


def test_sanity_check_fails_on_wrong_level(cfg):
    ref = cfg["sanity_checks"]["jkse_2008"]
    s = _jkse_2008_like(ref)
    s.loc[pd.Timestamp(ref["peak"]["date"])] *= 1.05
    res = q.sanity_check_2008(s, ref)
    assert res.loc[res["cek"] == "Puncak 2008", "status"].item() == "FAIL"


# ---- kualitas Open ------------------------------------------------------------------
def test_open_quality_detects_prev_close_fill_and_decides(qcfg):
    df = make_ohlc(random_walk(start="2007-01-01", end="2020-12-31"))
    bad_years = df.index.year <= 2011
    df.loc[bad_years, "Open"] = df["Close"].shift(1)[bad_years]   # Yahoo isi Open = Close kemarin
    overall, per_year = q.open_quality(df)
    assert per_year.loc[2008, "open_eq_prev_close"] > 90
    assert per_year.loc[2015, "open_eq_prev_close"] == 0
    dec = q.open_module_decision(overall, per_year, qcfg["open_checks"], qcfg["min_years_open_module"])
    assert dec["mode"] == "gap_intraday_partial" and dec["start_year"] == 2012


def test_open_decision_full_and_fallback(qcfg):
    good = make_ohlc(random_walk(start="2007-01-01", end="2020-12-31"))
    o, py = q.open_quality(good)
    assert q.open_module_decision(o, py, qcfg["open_checks"], 5)["mode"] == "gap_intraday"

    bad = good.copy()
    bad.loc[bad.index.year >= 2018, ["Open", "High", "Low"]] = np.nan  # Open hilang di tahun terakhir
    o, py = q.open_quality(bad)
    assert q.open_module_decision(o, py, qcfg["open_checks"], 5)["mode"] == "close_to_close"


def test_summary_mentions_bad_ticks(qcfg):
    qual = pd.DataFrame([{"name": "USD/IDR", "ticker": "IDR=X", "status": "3 dugaan bad tick"}])
    moves = pd.DataFrame({"id": ["USDIDR"] * 3, "bad_tick": [True, True, True]})
    sanity = pd.DataFrame([{"cek": "x", "tanggal": "2008-01-09", "status": "PASS", "catatan": ""}])
    text = q.phase1_summary(qual, {"reason": "ok"}, sanity, {"reason": "ok"}, moves)
    assert "USDIDR (3)" in text
