"""Pipeline Phase 1-2 end-to-end dengan target sintetis lain (tanpa internet).

Urutan langkahnya sama dengan notebooks/01 dan 02; tes ini menjaga bahwa mengganti
project.target_id tidak membutuhkan perubahan kode.
"""
import json

import numpy as np
import pandas as pd

from src import preprocess as pp
from src import quality as q
from src.config import (load_config, output_dir, resolve_target, sanity_ref, series_table, target_ticker,
                        target_usd_id)
from tests.conftest import make_ohlc, random_walk


def _synthetic_cfg():
    raw = load_config()
    raw["targets"].append(dict(id="SYN", ticker="SYN.JK", name="Saham sintetis", category="stock",
                               session="target", transform="log"))
    return resolve_target(raw, "SYN")


def _synthetic_prices(table, cfg):
    prices = {}
    for k, m in enumerate(table.to_dict("records")):
        level = 10000 if m["category"] == "fx" else 100
        s = random_walk(start="2007-01-01", end="2021-12-31", sd=0.01 if m["category"] != "fx" else 0.003,
                        level=level, seed=10 + k)
        if m["category"] == "fx":
            s.loc[pd.bdate_range("2013-11-29", "2013-12-03")] = 9612.45     # run bad tick 3 hari
        df = make_ohlc(s, seed=k)
        if m["group"] == "target":
            df = df.drop(pd.to_datetime(cfg["project"]["suspended_days"]), errors="ignore")
            susp = pd.Timestamp("2008-10-09")
            df.loc[susp] = df.loc[pd.Timestamp("2008-10-08")].values      # Yahoo isi hari suspend
            df.loc[susp, "Volume"] = 0
            df = df.sort_index()
        prices[m["ticker"]] = df
    return prices


def test_phase1_and_phase2_run_with_other_target(tmp_path):
    cfg = _synthetic_cfg()
    qcfg = cfg["quality"]
    series = series_table(cfg)
    target_id = cfg["project"]["target_id"]
    assert target_id == "SYN" and target_ticker(cfg) == "SYN.JK"
    prices = _synthetic_prices(series, cfg)
    tgt = prices[target_ticker(cfg)]

    # ---- Phase 1 (urutan notebook 01)
    start = pd.Timestamp(cfg["project"]["start_date"])
    quality = q.quality_report(prices, series, start, qcfg)
    moves = q.moves_table(prices, series, qcfg)
    start_dec = q.decide_start(tgt["Close"], cfg["project"]["start_date"], cfg["project"]["fallback_start"],
                               qcfg["min_obs_per_year"], qcfg["start_tolerance_days"], label="SYN.JK")
    ref = sanity_ref(cfg)
    sanity = q.sanity_check_2008(tgt["Close"], ref) if ref else None
    open_overall, open_per_year = q.open_quality(tgt)
    open_dec = q.open_module_decision(open_overall, open_per_year, qcfg["open_checks"], qcfg["min_years_open_module"])
    out1 = output_dir(cfg, "reports", "phase1", root=tmp_path)
    q.save_phase1_outputs(out1, quality, moves, sanity, open_per_year, start_dec, open_dec, target_id)
    assert "SYN" in quality["id"].tolist() and sanity is None
    assert "dilewati" in q.phase1_summary(quality, start_dec, sanity, open_dec, moves, "SYN.JK")
    assert out1 == tmp_path / "reports" / "targets" / "SYN" / "phase1"

    # ---- Phase 2 (urutan notebook 02)
    dec1 = json.loads((out1 / "decisions.json").read_text())
    dec1["return_mode"] = "close_to_close"                      # data sintetis: Open bersih
    (out1 / "decisions.json").write_text(json.dumps(dec1))
    dec1 = pp.load_phase1_decisions(out1 / "decisions.json")
    assert dec1["target_id"] == "SYN"
    clean, log, flags = pp.clean_prices(prices, series, cfg)
    assert pd.Timestamp("2008-10-09") in set(log.loc[log["id"] == "SYN", "Date"])      # suspend -> libur
    assert len(log[log["id"] == cfg["usdidr_retest"]["id"]]) == 3                       # bad tick FX
    transforms = dict(zip(series["id"], series["transform"]))
    usd_id = target_usd_id(cfg)
    usd = pp.target_in_usd(clean, target_id, cfg["target_usd"]["fx_id"], usd_id)
    daily = pp.pair_daily_returns(clean, target_id, transforms, flags)
    weekly = pp.weekly_returns(clean.join(usd, how="left"), {**transforms, usd_id: "log"})
    assert target_id != cfg["usdidr_retest"]["target_id"]
    decision = pp.retest_skipped(target_id, cfg["usdidr_retest"]["target_id"])
    out2 = output_dir(cfg, "reports", "phase2", root=tmp_path)
    paths = pp.save_phase2_outputs(out2, log, pd.DataFrame(), flags[flags.any(axis=1)], pp.pair_coverage(daily),
                                   moves[moves["category"] != "fx"], decision, {"target_id": target_id})
    pp.save_processed(output_dir(cfg, "data/processed", root=tmp_path), {"daily_pair_returns": daily})
    assert all(p.exists() for p in paths)
    assert set(daily["id"]) == set(series["id"]) - {target_id}
    assert usd_id in weekly and np.isfinite(weekly[usd_id]).sum() > 500
    assert "SYN" in pp.phase2_summary(log, flags, decision, pp.pair_coverage(daily), moves, "USDIDR", 0)
