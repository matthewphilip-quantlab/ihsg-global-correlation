import json

import numpy as np
import pandas as pd
import pytest

from src import bigcap as bc
from src import data_loader as dl
from src import preprocess as pp
from src.config import load_config, output_dir, resolve_target, series_table
from tests.conftest import make_ohlc, random_walk


# ---- aturan dividen besar ------------------------------------------------------------------------
def _actions():
    idx = pd.bdate_range("2023-06-19", periods=8)
    close = pd.Series([3790, 3850, 3850, 3780, 3700, 3150, 2680, 2720.0], index=idx)
    div = pd.Series(0.0, index=idx)
    div.iloc[5] = 1094.05            # 29,6% dari close sebelumnya -> kena
    div.iloc[2] = 100.0              # 2,6% -> tidak kena
    return pd.DataFrame({"close_raw": close, "dividend": div}), idx


def test_large_dividend_days_threshold():
    act, idx = _actions()
    days = pp.large_dividend_days(act, 0.10)
    assert list(days.index) == [idx[5]] and days["rasio"].iloc[0] == pytest.approx(1094.05 / 3700)
    log = pp.dividend_log_rows(days, "PTBA", "PTBA.JK", 0.10)
    assert list(log.columns) == pp.LOG_COLS and "dividen besar" in log["reason"].iloc[0]


def test_mask_dividend_daily_and_weekly():
    _, idx = _actions()
    daily = pd.DataFrame({"Date": idx, "id": "X", "r_target": 0.01, "r_x": 0.02})
    m = pp.mask_dividend_daily(daily, pd.DatetimeIndex([idx[5]]))
    assert m["r_target"].isna().sum() == 1 and m["r_x"].notna().all()
    weekly = pd.DataFrame({"T": [0.1, 0.2], "X": [0.3, 0.4]}, index=pd.to_datetime(["2023-06-23", "2023-06-30"]))
    w = pp.mask_dividend_weekly(weekly, "T", pd.DatetimeIndex([idx[5]]))   # Senin 26 Jun -> minggu 30 Jun
    assert np.isnan(w.loc["2023-06-30", "T"]) and w.loc["2023-06-23", "T"] == 0.1 and w["X"].notna().all()


def test_load_actions_uses_cache(tmp_path, monkeypatch):
    calls = []
    act, _ = _actions()

    def fake(ticker, start, end=None):
        calls.append(ticker)
        return act
    monkeypatch.setattr(dl, "download_actions", fake)
    a = dl.load_actions(["PTBA.JK"], "2023-01-01", cache_dir=tmp_path)
    b = dl.load_actions(["PTBA.JK"], "2023-01-01", cache_dir=tmp_path)
    assert calls == ["PTBA.JK"] and a["PTBA.JK"].equals(b["PTBA.JK"])
    assert "PTBA.JK#actions" in json.loads((tmp_path / "_manifest.json").read_text())


# ---- pipeline end-to-end dengan saham sintetis ---------------------------------------------------------
def _synthetic_stock_cfg():
    raw = load_config()
    raw["targets"].append(dict(id="SYN", ticker="SYN.JK", name="Saham sintetis", category="stock",
                               session="target", transform="log"))
    cfg = resolve_target(raw, "SYN")
    cfg["analysis"]["asymmetric"] |= {"n_boot": 60, "n_mc": 20_000}
    return cfg


def _synthetic_prices(cfg):
    series = series_table(cfg)
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2007-01-01", "2023-06-30")
    common = rng.normal(0, 0.01, len(idx))
    prices = {}
    for k, m in enumerate(series.to_dict("records")):
        lvl = {"fx": 10000, "sentiment": 20}.get(m["category"], 100)
        eps = rng.normal(0, 0.01, len(idx))
        r = (0.6 * common + 0.8 * eps) if m["category"] != "fx" else (-0.2 * common + 0.4 * eps)
        prices[m["ticker"]] = make_ohlc(pd.Series(lvl * np.exp(np.cumsum(r)), index=idx), seed=k)
    market = pd.Series(5000 * np.exp(np.cumsum(0.8 * common + 0.5 * rng.normal(0, 0.01, len(idx)))), index=idx)
    return prices, market, idx


def test_bigcap_pipeline_end_to_end(tmp_path):
    cfg = _synthetic_stock_cfg()
    prices, market, idx = _synthetic_prices(cfg)
    ex = idx[3000]
    close_raw = prices["SYN.JK"]["Close"]
    actions = pd.DataFrame({"close_raw": close_raw, "dividend": 0.0})
    actions.loc[ex, "dividend"] = 0.3 * close_raw.loc[:ex].iloc[-2]

    p1 = bc.run_phase1(cfg, prices, root=tmp_path)
    assert (tmp_path / "reports" / "targets" / "SYN" / "phase1" / "decisions.json").exists()
    assert not p1["start"]["used_fallback"]
    p2 = bc.run_phase2(cfg, prices, actions, "2007-01-01", root=tmp_path)
    assert list(p2["div_days"].index) == [ex]
    assert (p2["log"]["Date"] == ex).any() and "dividen" in p2["log"].loc[p2["log"]["Date"] == ex, "reason"].iloc[0]
    assert (output_dir(cfg, "data/processed", root=tmp_path) / "prices_clean.parquet").exists()

    panel = bc.stock_panel(p2["clean"], p2["flags"], market, cfg["project"]["primary_target"])
    res = bc.stock_analysis(cfg, panel, p2["div_days"], use_div_rule=True)
    sens = bc.stock_analysis(cfg, panel, p2["div_days"], use_div_rule=False)
    row = res["row"]
    assert row["n_hari_dividen_besar"] == 1 and row["aturan_dividen"] and not sens["row"]["aturan_dividen"]
    assert 0.3 < row["beta_ihsg_2010_2019"] < 1.5 and row["beta_usdidr_2010_2019"] < 0
    assert set(res["betas"]["faktor"]) == {"JKSE", "USDIDR", "EEM"} and len(res["event_study"]) == 6
    assert np.isfinite(row["garch_gamma_is"]) and 0 <= row["garch_dm_qlike_p"] <= 1
    table, champs = bc.comparison_table([row, {**sens["row"], "id": "SYN_tanpa_aturan"}])
    assert len(table) == 2 and len(champs) == 5 and set(champs["saham"]) <= {"SYN", "SYN_tanpa_aturan"}


def test_bigcap_summary_and_plot():
    import matplotlib
    matplotlib.use("Agg")
    from src import plots
    t = pd.DataFrame({"garch_dm_qlike_p": [0.01, 0.5], "garch_dm_qlike_selisih": [-0.1, 0.1], "lead_p_dpremium": [0.2, 0.01],
                      "beta_eem_2010_2019": [0.5, 0.7], "beta_eem_2020+": [0.4, 0.6]}, index=["A", "B"])
    champs = pd.DataFrame({"saham": ["A"], "nilai": [-0.5]}, index=["paling sensitif ke rupiah"])
    txt = bc.bigcap_summary(t, champs, {"A": "Saham A"})
    assert "Saham A (-0.500)" in txt and "VIX membantu" in txt and "B" in txt
    assert plots.plot_bigcap_betas(t, {"EEM": ("beta_eem_2010_2019", "beta_eem_2020+")}, {}, "tes") is not None
