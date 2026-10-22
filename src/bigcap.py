"""Demonstrasi pipeline untuk target saham big cap (Phase 7).

Target dipilih lewat load_config(target_id=...). Seri Core, aturan, dan periode SAMA dengan IHSG; tidak ada seleksi
ulang. Aturan dividen besar (README poin 9) berlaku untuk target berkategori stock, dengan versi tanpa aturan
sebagai sensitivitas.

Isi modul:
- run_phase1 / run_phase2       : data quality & cleaning per target (output di reports/targets/<id>/)
- stock_panel                   : panel harga bersih target + Core + alat bantu + IHSG (sebagai faktor pasar)
- beta_table / asymmetry_vs     : beta mingguan HAC satu faktor; korelasi asimetris vs EEM
- lead_table / crisis_medians / garch_vix : lead harian, event study, GARCH-X dengan VIX (IS & OOS)
- stock_analysis / comparison_table
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import analysis as an
from . import leadlag as ll
from . import oos
from . import preprocess as pp
from . import quality as q
from . import regime as rg
from . import volatility as vol
from .config import output_dir, sanity_ref, series_table, target_ticker


# ---- Phase 1 & 2 per target ------------------------------------------------------------------------
def run_phase1(cfg: dict, prices: dict[str, pd.DataFrame], root=None) -> dict:
    """Langkah notebook 01 untuk target aktif; hasil disimpan ke reports/targets/<id>/phase1/."""
    qcfg, series = cfg["quality"], series_table(cfg)
    tid, label = cfg["project"]["target_id"], target_ticker(cfg)
    tgt = prices[label]
    quality = q.quality_report(prices, series, pd.Timestamp(cfg["project"]["start_date"]), qcfg)
    moves = q.moves_table(prices, series, qcfg)
    start = q.decide_start(tgt["Close"], cfg["project"]["start_date"], cfg["project"]["fallback_start"],
                           qcfg["min_obs_per_year"], qcfg["start_tolerance_days"], label=label)
    ref = sanity_ref(cfg)
    sanity = q.sanity_check_2008(tgt["Close"], ref) if ref else None
    overall, per_year = q.open_quality(tgt)
    open_dec = q.open_module_decision(overall, per_year, qcfg["open_checks"], qcfg["min_years_open_module"])
    kw = {} if root is None else {"root": root}
    q.save_phase1_outputs(output_dir(cfg, "reports", "phase1", **kw), quality, moves, sanity, per_year, start, open_dec, tid)
    return {"quality": quality, "start": start, "open": open_dec, "moves": moves}


def run_phase2(cfg: dict, prices: dict[str, pd.DataFrame], actions: pd.DataFrame | None, analysis_start, root=None) -> dict:
    """Langkah notebook 02 untuk target aktif + aturan dividen besar. Uji ulang USD/IDR tidak dijalankan ulang."""
    series = series_table(cfg)
    tid = cfg["project"]["target_id"]
    meta = series.set_index("id").loc[tid]
    clean, log, flags = pp.clean_prices(prices, series, cfg)
    frac = cfg["quality"]["large_dividend_frac"]
    div_days = (pp.large_dividend_days(actions, frac) if meta["category"] == "stock" and actions is not None
                else pd.DataFrame(columns=["dividen", "close_sebelumnya", "rasio"]))
    if len(div_days):
        log = pd.concat([log, pp.dividend_log_rows(div_days, tid, meta["ticker"], frac)], ignore_index=True)
    transforms = dict(zip(series["id"], series["transform"]))
    daily = pp.pair_daily_returns(clean, tid, transforms, flags)
    weekly = pp.weekly_returns(clean, transforms)
    kw = {} if root is None else {"root": root}
    decision = pp.retest_skipped(tid, cfg["usdidr_retest"]["target_id"])
    moves = q.moves_table(prices, series, cfg["quality"])
    pp.save_phase2_outputs(output_dir(cfg, "reports", "phase2", **kw), log, pd.DataFrame(), flags[flags.any(axis=1)],
                           pp.pair_coverage(daily), moves[moves["category"] != "fx"], decision,
                           {"target_id": tid, "analysis_start": analysis_start, "return_mode": "close_to_close",
                            "dividen_besar": [d.strftime("%Y-%m-%d") for d in div_days.index]})
    pp.save_processed(output_dir(cfg, "data/processed", **kw),
                      {"prices_clean": clean, "stale_quote_flags": flags, "daily_pair_returns": daily, "weekly_returns": weekly})
    return {"clean": clean, "log": log, "flags": flags, "daily": daily, "weekly": weekly, "div_days": div_days}


def stock_panel(clean: pd.DataFrame, flags: pd.DataFrame, market: pd.Series, market_id: str) -> pd.DataFrame:
    """Harga bersih (kuotasi basi dibuang, keputusan Phase 4) + IHSG bersih sebagai faktor pasar."""
    return ll.drop_flagged(clean, flags).join(market.rename(market_id), how="left")


# ---- Analisis A2-A6 ---------------------------------------------------------------------------------
def beta_table(weekly: pd.DataFrame, tid: str, factors: list[str], periods: dict[str, tuple]) -> pd.DataFrame:
    """Beta mingguan HAC satu faktor (tiap faktor terpisah) per periode."""
    rows = []
    for per, (a, b) in periods.items():
        for f in factors:
            coef, model = an.factor_regression(weekly, tid, [f], a, b)
            rows.append({"periode": per, "faktor": f, "beta": coef.loc[f, "beta"], "ci_lo": coef.loc[f, "ci_lo"],
                         "ci_hi": coef.loc[f, "ci_hi"], "p": coef.loc[f, "p"], "r2": model["r2"], "n": model["n_minggu"]})
    return pd.DataFrame(rows)


def asymmetry_vs(weekly: pd.DataFrame, tid: str, cond_id: str, acfg: dict) -> pd.DataFrame:
    """Korelasi target-cond di minggu cond turun vs naik (tanda & exceedance), dengan pembanding normal."""
    return an.asymmetric_table(weekly, tid, [cond_id], [], None, None, acfg)


def lead_table(panel: pd.DataFrame, tid: str, market_id: str, fx_id: str, eido_id: str, eem_id: str,
               transforms: dict, end: str, start: str, div_dates: pd.DatetimeIndex | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return saham t+1 ~ EEM(t), Δpremium EIDO(t) (premium EIDO vs IHSG dalam USD), in-sample start..end, HAC.
    Return saham yang terealisasi di ex-date dividen besar dibuang kalau div_dates diberi."""
    feats = oos.feature_frame(panel, market_id, [eem_id], transforms, fx_id=fx_id, eido_id=eido_id)
    tgt = oos.next_day_target(panel[tid])
    if div_dates is not None and len(div_dates):
        tgt.loc[tgt["target_date"].isin(div_dates), "r_next"] = np.nan
    f = tgt[["r_next"]].join(feats[[f"r_{eem_id}", "d_premium"]], how="inner").loc[start:end]
    models = {"EEM": [f"r_{eem_id}"], "Δpremium": ["d_premium"], "EEM + Δpremium": [f"r_{eem_id}", "d_premium"]}
    return ll.nested_regressions(f, "r_next", models, "EEM")


def crisis_medians(panel: pd.DataFrame, tid: str, rcfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Event study 6 episode utama (median), tanpa episode deskriptif."""
    empty = pd.DataFrame(columns=["Date", "id", "r_target", "r_x"])
    es = rg.event_study(panel, empty, tid, [], rcfg["episodes"], tuple(rcfg["window_days"]), rcfg["pre_days"])
    med = rg.summarize_episodes(es, [], ["max_drawdown", "return_kumulatif_akhir", "hari_ke_titik_terendah"]).loc[tid]
    return es, med


def garch_vix(panel: pd.DataFrame, tid: str, vix_id: str, train_start: str, is_end: str, oos_start: str,
              div_dates: pd.DatetimeIndex | None) -> dict:
    """GARCH-X dengan VIX untuk target: gamma in-sample (train_start..is_end) + uji LR gamma = 0, lalu
    walk-forward OOS seperti Phase 6 (QLIKE/MSE, DM). Return ex-date dividen besar dibuang kalau div_dates diberi."""
    ret = (np.log(panel[tid].dropna()).diff() * 100).dropna()
    if div_dates is not None and len(div_dates):
        ret = ret.drop(index=ret.index.intersection(div_dates))
    x = oos.own_returns(panel[vix_id]).reindex(ret.index).fillna(0.0)
    r_is, x_is = ret.loc[train_start:is_end].to_numpy(), x.loc[train_start:is_end].to_numpy()
    fx, f0 = vol.fit_garchx(r_is, x_is), vol.fit_garchx(r_is, x_is, gamma_fixed=0.0)
    lr = 2 * (fx["loglik"] - f0["loglik"])
    g_base = vol.garch_walk_forward(ret, x, train_start, oos_start, use_x=False)
    g_x = vol.garch_walk_forward(ret, x, train_start, oos_start, use_x=True)
    cmp = vol.compare_losses(g_x, g_base, "r2")
    return {"gamma_is": float(fx["params"][4]), "lr_p_is": float(stats.chi2.sf(max(lr, 0), 1)),
            "qlike_x": cmp.loc["QLIKE", "dengan_vix"], "qlike_base": cmp.loc["QLIKE", "tanpa_vix"],
            "dm_qlike_p": cmp.loc["QLIKE", "p"], "dm_qlike_selisih": cmp.loc["QLIKE", "selisih_loss_rata"],
            "dm_mse_p": cmp.loc["MSE", "p"], "n_oos": int(cmp.loc["QLIKE", "n"])}


def stock_analysis(cfg: dict, panel: pd.DataFrame, div_days: pd.DataFrame, use_div_rule: bool) -> dict:
    """A2-A6 untuk target aktif. Mengembalikan tabel per analisis dan satu baris ringkasan."""
    tid, market_id = cfg["project"]["target_id"], cfg["project"]["primary_target"]
    fx_id, eido_id, vix_id = cfg["usdidr_retest"]["id"], cfg["oos"]["eido_id"], cfg["oos"]["vix_id"]
    bcfg = cfg["bigcap"]
    eem_id = bcfg["lead_id"]
    series = series_table(cfg)
    transforms = dict(zip(series["id"], series["transform"])) | {market_id: "log"}
    win = cfg["project"]["screening_window"]
    periods = {"2010-2019": (win[0], win[1]), "2020+": (cfg["oos"]["oos_start"], None)}
    dates = pd.DatetimeIndex(div_days.index) if use_div_rule else pd.DatetimeIndex([])
    weekly = pp.weekly_returns(panel, transforms)
    if len(dates):
        weekly = pp.mask_dividend_weekly(weekly, tid, dates)
    betas = beta_table(weekly, tid, [market_id] + list(bcfg["factors"]), periods)
    asym = asymmetry_vs(weekly, tid, bcfg["asym_id"], cfg["analysis"]["asymmetric"])
    lead_coef, lead_models = lead_table(panel, tid, market_id, fx_id, eido_id, eem_id, transforms,
                                        win[1], cfg["oos"]["train_start"], dates)
    es, med = crisis_medians(panel, tid, cfg["regime"])
    g = garch_vix(panel, tid, vix_id, cfg["oos"]["train_start"], win[1], cfg["oos"]["oos_start"], dates)
    b = betas.set_index(["periode", "faktor"])
    a = asym[asym["mode"] == "tanda"].iloc[0]
    lc = lead_coef.set_index(["model", "variabel"])
    row = {"id": tid,
           "beta_ihsg_2010_2019": b.loc[("2010-2019", market_id), "beta"], "beta_ihsg_2020+": b.loc[("2020+", market_id), "beta"],
           "beta_usdidr_2010_2019": b.loc[("2010-2019", fx_id), "beta"], "beta_usdidr_2020+": b.loc[("2020+", fx_id), "beta"],
           "beta_eem_2010_2019": b.loc[("2010-2019", bcfg["asym_id"]), "beta"], "beta_eem_2020+": b.loc[("2020+", bcfg["asym_id"]), "beta"],
           "r_turun_eem": a["r_turun"], "r_naik_eem": a["r_naik"], "asimetri_berlebih": a["asimetri_berlebih"],
           "asimetri_ci_lo": a["ci_lo_boot"], "asimetri_ci_hi": a["ci_hi_boot"],
           "lead_beta_eem": lc.loc[("EEM", f"r_{eem_id}"), "beta"], "lead_beta_dpremium": lc.loc[("Δpremium", "d_premium"), "beta"],
           "lead_tambahan_r2_dpremium": lead_models.loc["EEM + Δpremium", "tambahan_r2_vs_base"],
           "lead_p_dpremium": lead_models.loc["EEM + Δpremium", "p_tambahan"],
           "krisis_drawdown_median": med["max_drawdown"], "krisis_return60_median": med["return_kumulatif_akhir"],
           **{f"garch_{k}": v for k, v in g.items()},
           "n_hari_dividen_besar": int(len(div_days)), "aturan_dividen": use_div_rule}
    return {"row": row, "betas": betas, "asym": asym, "lead_coef": lead_coef, "lead_models": lead_models,
            "event_study": es, "garch": g}


def comparison_table(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabel ringkas + 'juara' tiap dimensi (paling sensitif rupiah = beta USD/IDR paling negatif, dst.)."""
    t = pd.DataFrame(rows).set_index("id")
    champions = pd.DataFrame({
        "paling sensitif ke rupiah (beta USD/IDR paling negatif, 2010-2019)": [t["beta_usdidr_2010_2019"].idxmin(), t["beta_usdidr_2010_2019"].min()],
        "paling sensitif ke EM (beta EEM terbesar, 2010-2019)": [t["beta_eem_2010_2019"].idxmax(), t["beta_eem_2010_2019"].max()],
        "paling asimetris vs EEM (asimetri berlebih terbesar)": [t["asimetri_berlebih"].idxmax(), t["asimetri_berlebih"].max()],
        "paling dalam saat krisis (median drawdown)": [t["krisis_drawdown_median"].idxmin(), t["krisis_drawdown_median"].min()],
        "paling responsif ke VIX (gamma GARCH-X in-sample)": [t["garch_gamma_is"].idxmax(), t["garch_gamma_is"].max()],
    }, index=["saham", "nilai"]).T
    return t, champions


def bigcap_summary(t: pd.DataFrame, champs: pd.DataFrame, names: dict, alpha: float = 0.05) -> str:
    """Ringkasan big cap dalam bahasa sederhana (DESKRIPTIF; bukan rekomendasi saham)."""
    n = lambda i: names.get(i, i)  # noqa: E731
    L = ["### Ringkasan big cap (demonstrasi pipeline; deskriptif, bukan rekomendasi)", ""]
    for dim, r in champs.iterrows():
        L.append(f"- **{dim}:** {n(r.saham)} ({r.nilai:+.3f}).")
    vix = t[(t["garch_dm_qlike_p"] < alpha) & (t["garch_dm_qlike_selisih"] < 0)].index
    L.append("- **VIX membantu prediksi volatilitas OOS (DM QLIKE, α 5%):** "
             + (", ".join(n(i) for i in vix) if len(vix) else "tidak ada") + f" dari {len(t)} saham.")
    lead = t[t["lead_p_dpremium"] < alpha].index
    L.append("- **Δpremium EIDO menambah daya jelas lead di atas EEM (in-sample 2010-2019):** "
             + (", ".join(n(i) for i in lead) if len(lead) else "tidak ada") + ".")
    return "\n".join(L)
