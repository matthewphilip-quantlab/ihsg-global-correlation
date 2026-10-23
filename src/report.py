"""Bahan Lampiran Teknis (PDF): grafik dibuat ulang dari reports/ dan data/processed, tanpa analisis baru.

Isi modul:
- fig_* / FIGURES        : pembangun grafik per phase (memanggil src/plots.py); dipakai sel notebook bertag "report"
                           dan scripts/build_notebook_report.py
- save_vol_predictions   : seri prakiraan GARCH-X vs GARCH (OOS) untuk grafik; dihitung ulang lalu DIVERIFIKASI sama
                           dengan reports/phase6/vol_garch_vs_garchx.csv sebelum disimpan
- csv / js               : pembaca reports/
- snippet / fmt_id / crosscheck : potongan kode, format angka gaya Indonesia, cek silang angka ke README
"""
from __future__ import annotations

import inspect
import json
import textwrap
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import leadlag as ll
from . import plots
from . import regime as rg
from . import volatility as vol
from . import oos
from .config import PROJECT_ROOT, load_config, output_dir, sanity_ref, series_table

REPORTS = PROJECT_ROOT / "reports"


# ---- Pembaca ------------------------------------------------------------------------------------------
def csv(rel: str, **kw) -> pd.DataFrame:
    return pd.read_csv(REPORTS / rel, **kw)


def js(rel: str) -> dict:
    return json.loads((REPORTS / rel).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _cfg() -> dict:
    return load_config()


@lru_cache(maxsize=1)
def names() -> dict:
    cfg = _cfg()
    t = series_table(cfg, groups=("target", "core", "tools", "nanti"))
    out = dict(zip(t["id"], t["name"]))
    out.update({t_["id"]: t_["name"] for t_ in cfg["targets"]})
    out.update({"GOLD_IDR": "Emas (dalam rupiah)", "USDIDR_HOLD": "Memegang USD"})
    return out


@lru_cache(maxsize=1)
def processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    proc = output_dir(_cfg(), "data/processed")
    return pd.read_parquet(proc / "prices_clean.parquet"), pd.read_parquet(proc / "stale_quote_flags.parquet")


def _windows():
    cfg = _cfg()
    prices, _ = processed()
    td = prices[cfg["project"]["target_id"]].dropna().index
    rc = cfg["regime"]
    return td, {k: rg.episode_window(td, a, tuple(rc["window_days"])) for k, a in (rc["episodes"] | rc["descriptive_only"]).items()}


# ---- Phase 1 --------------------------------------------------------------------------------------------
def fig_availability():
    cfg = _cfg()
    prices, _ = processed()
    ids = [cfg["project"]["target_id"]] + [s["id"] for s in cfg["core"]] + [s["id"] for s in cfg["tools"]]
    counts = prices[ids].notna().groupby(prices.index.year).sum().T
    counts = counts.where(counts > 0)
    counts.index = [names().get(i, i) for i in counts.index]
    return plots.plot_availability(counts, "Jumlah hari berharga per seri per tahun (panel data/processed; 2026 s/d 21 Okt)")


def fig_open_quality():
    per_year = csv("phase1/open_quality_by_year.csv", index_col=0)
    return plots.plot_open_quality(per_year, _cfg()["quality"]["open_checks"], "^JKSE")


def fig_jkse_2008():
    prices, _ = processed()
    return plots.plot_jkse_2008(prices[_cfg()["project"]["target_id"]], sanity_ref(_cfg()))


# ---- Phase 2 --------------------------------------------------------------------------------------------
def fig_fx_cleaning():
    cfg = _cfg()
    fx = cfg["usdidr_retest"]["id"]
    prices, _ = processed()
    clean = prices[fx]
    log = csv("phase2/cleaning_log.csv", parse_dates=["Date"])
    removed = log[log["id"] == fx].set_index("Date")["close"]
    raw = clean.copy()
    raw.loc[removed.index] = removed.to_numpy()            # harga mentah = bersih + harga yang dibuang (cleaning log)
    return plots.plot_fx_cleaning(raw, clean, "USD/IDR 2010-2014: sebelum vs sesudah cleaning (Amandemen 1)",
                                  "2010-01-01", "2014-12-31")


def jkse_daily_returns() -> pd.Series:
    prices, _ = processed()
    return np.log(prices[_cfg()["project"]["target_id"]].dropna()).diff().dropna()


def fig_return_distribution():
    return plots.plot_return_distribution(jkse_daily_returns(), "Return log harian IHSG 2007-2026: ekor gemuk (fat tails)")


# ---- Phase 3 --------------------------------------------------------------------------------------------
def fig_corr_heatmap():
    m = csv("phase3/matriks_korelasi_mingguan_screening.csv", index_col=0)
    return plots.plot_corr_heatmap(m.rename(index=names(), columns=names()), "Korelasi mingguan IHSG & seri Core, 2010-2019")


def fig_beta_forest():
    coef = csv("phase3/regresi_faktor_screening.csv", index_col=0)
    one = csv("phase3/beta_satu_faktor_screening.csv", index_col=0)
    return plots.plot_beta_forest(coef, names(), "Beta mingguan IHSG terhadap seri Core, 2010-2019", single=one)


def fig_asymmetric():
    a = csv("phase3/asimetris_penuh.csv")
    cond = _cfg()["analysis"]["asymmetric"]["condition_ids"][0]
    a = a[(a["kondisi"] == cond) & (a["mode"] == "tanda")]
    return plots.plot_asymmetric(a, names(), f"Korelasi dengan IHSG saat {names()[cond]} turun vs naik (mingguan, 2007-2026)")


# ---- Phase 4 --------------------------------------------------------------------------------------------
def fig_ccf():
    ccf = csv("phase4/ccf.csv")
    t = series_table(_cfg())
    return plots.plot_ccf_heatmap(ccf, names(), dict(zip(t["id"], t["session"])), "Korelasi IHSG(t+k) vs seri(t), harian 2010-2019")


def eido_premium() -> pd.Series:
    cfg = _cfg()
    prices, flags = processed()
    start, end = cfg["project"]["screening_window"]
    close = ll.drop_flagged(ll.in_sample(prices.loc[start:], end), ll.in_sample(flags.loc[start:], end))
    t = series_table(cfg)
    ef = ll.eido_frame(close, cfg["project"]["target_id"], cfg["usdidr_retest"]["id"], cfg["analysis"]["leadlag"]["eido_id"],
                       cfg["analysis"]["leadlag"]["eido_compare_ids"], dict(zip(t["id"], t["transform"])))
    return ef["premium"]


def fig_eido_premium():
    return plots.plot_premium(eido_premium(), "Premium EIDO terhadap IHSG dalam USD, in-sample 2010-2019")


# ---- Phase 5 --------------------------------------------------------------------------------------------
def fig_rolling(ids=("HSI", "SPX", "EEM", "USDIDR")):
    r52 = csv("phase5/rolling_52.csv", index_col=0, parse_dates=True)
    r156 = csv("phase5/rolling_156.csv", index_col=0, parse_dates=True)
    _, wins = _windows()
    return plots.plot_rolling_corr({52: r52[list(ids)], 156: r156[list(ids)]}, names(), [(w.start, w.end) for w in wins.values()],
                                   "Korelasi mingguan bergulir IHSG (arsir = jendela krisis)")


def fig_event_paths():
    cfg = _cfg()
    prices, flags = processed()
    close = ll.drop_flagged(prices, flags)
    td, wins = _windows()
    rc = cfg["regime"]
    paths = {k: rg.event_paths(close, [cfg["project"]["target_id"]] + rc["event_ids"], w, td) for k, w in wins.items()}
    return plots.plot_event_paths(paths, names(), "Harga di sekitar jangkar krisis (close sebelum hari 0 = 100)",
                                  notes={k: "deskriptif, terpotong" for k in rc["descriptive_only"]})


def fig_quantile():
    qr = {"2010-2019": csv("phase5/brent_kuantil_2010_2019.csv"), "seluruh periode": csv("phase5/brent_kuantil_penuh.csv")}
    for k in qr:
        qr[k]["kuantil"] = qr[k]["kuantil"].map(lambda v: v if v == "OLS" else float(v))
    return plots.plot_quantile_coefs(qr, _cfg()["regime"]["brent_id"],
                                     "Koefisien Brent per kuantil return IHSG (kontrol Tembaga & EEM)")


def haven_medians() -> pd.Series:
    h = csv("phase5/pelindung_per_episode.csv")
    h = h[~h["episode"].astype(str).isin(list(_cfg()["regime"]["descriptive_only"])) & (h["satuan"] == "%")]
    return h.groupby("id")["ke_titik_terendah_target"].median()


def fig_havens():
    n = names() | {_cfg()["usdidr_retest"]["id"]: "Memegang USD (USD/IDR)"}
    return plots.plot_haven_bars(haven_medians(), n, "Median perubahan sampai titik terendah IHSG, 6 episode krisis",
                                 highlight=_cfg()["project"]["target_id"])


# ---- Phase 6 --------------------------------------------------------------------------------------------
def fig_walkforward():
    p = csv("phase6/prediksi_arah.csv", parse_dates=["refit"])
    return plots.plot_walkforward_scheme(pd.DatetimeIndex(p["refit"]), _cfg()["oos"]["train_start"],
                                         "Walk-forward: latih expanding sejak 2010, refit & uji per bulan (contoh tiap 12 bulan)")


def fig_calibration():
    cal = csv("phase6/arah_kalibrasi_utama.csv", index_col=0)
    return plots.plot_calibration({"Logit utama (OOS)": cal}, "Kalibrasi prediksi arah IHSG besok, OOS 2020-2026")


VOL_PRED = "phase6/vol_prediksi_garch.csv"


def save_vol_predictions(tol: float = 1e-3) -> dict:
    """Hitung ulang walk-forward GARCH/GARCH-X (sama dengan notebook 06), cek QLIKE & MSE identik dengan
    reports/phase6/vol_garch_vs_garchx.csv, lalu simpan seri prakiraannya untuk grafik. Gagal = tidak disimpan."""
    cfg = _cfg()
    prices, flags = processed()
    close = ll.drop_flagged(prices, flags)
    ret = (np.log(close[cfg["project"]["target_id"]].dropna()).diff() * 100).dropna()
    x = oos.own_returns(close[cfg["oos"]["vix_id"]]).reindex(ret.index).fillna(0.0)
    o = cfg["oos"]
    g0 = vol.garch_walk_forward(ret, x, o["train_start"], o["oos_start"], use_x=False)
    g1 = vol.garch_walk_forward(ret, x, o["train_start"], o["oos_start"], use_x=True)
    cmp = vol.compare_losses(g1, g0, "r2")
    ref = csv("phase6/vol_garch_vs_garchx.csv", index_col=0)
    diff = (cmp[["dengan_vix", "tanpa_vix"]] - ref[["dengan_vix", "tanpa_vix"]]).abs().max().max()
    if not diff < tol * ref[["dengan_vix", "tanpa_vix"]].abs().max().max():
        raise ValueError(f"hasil hitung ulang berbeda dari reports/phase6 (selisih maks {diff:.6f}); tidak disimpan")
    out = pd.DataFrame({"r2": g1["r2"], "s2_vix": g1["s2"], "s2_tanpa_vix": g0["s2"]})
    out.to_csv(REPORTS / VOL_PRED)
    return {"selisih_maks_vs_reports": float(diff), "n": int(len(out)), "qlike": cmp.loc["QLIKE", ["dengan_vix", "tanpa_vix"]].round(4).to_dict()}


def fig_vol_forecast():
    df = csv(VOL_PRED, index_col=0, parse_dates=True)
    return plots.plot_vol_forecast(df, "Volatilitas IHSG OOS: |return| harian vs prakiraan sd GARCH-X & GARCH")


def fig_placebo():
    plac = csv("phase6/placebo_target_acak.csv")
    m = csv("phase6/arah_metrik_utama.csv", index_col=0)
    real = {"selisih_log_loss": float(m.loc["utama", "log_loss"] - m.loc["baseline", "log_loss"]),
            "hit_rate": float(m.loc["utama", "hit_rate"])}
    return plots.plot_placebo(plac, real, "Uji placebo: 10 model dengan target diacak vs model asli (OOS)")


# ---- Phase 7 --------------------------------------------------------------------------------------------
def fig_bigcap_betas():
    t = csv("phase7/bigcap_perbandingan.csv", index_col=0)
    return plots.plot_bigcap_betas(t, {"Beta pasar (IHSG)": ("beta_ihsg_2010_2019", "beta_ihsg_2020+"),
                                       "Beta rupiah (USD/IDR)": ("beta_usdidr_2010_2019", "beta_usdidr_2020+"),
                                       "Beta EM (EEM)": ("beta_eem_2010_2019", "beta_eem_2020+")},
                                   names(), "Beta mingguan satu faktor big cap: 2010-2019 vs 2020+")


def fig_bigcap_drawdown():
    t = csv("phase7/bigcap_perbandingan.csv", index_col=0)
    ref = csv("phase5/event_study_median.csv", index_col=0).loc[_cfg()["project"]["target_id"], "max_drawdown"]
    return plots.plot_bigcap_drawdown(t["krisis_drawdown_median"], names(), "Median drawdown big cap, 6 episode krisis",
                                      reference=("IHSG", float(ref)))


FIGURES = {
    1: ["availability", "open_quality", "jkse_2008"],
    2: ["fx_cleaning", "return_distribution"],
    3: ["corr_heatmap", "beta_forest", "asymmetric"],
    4: ["ccf", "eido_premium"],
    5: ["rolling", "event_paths", "quantile", "havens"],
    6: ["walkforward", "calibration", "vol_forecast", "placebo"],
    7: ["bigcap_betas", "bigcap_drawdown"],
}


def figure(key: str):
    return globals()[f"fig_{key}"]()


def show(*keys: str) -> None:
    """Untuk sel notebook bertag 'report': tampilkan grafik Lampiran Teknis."""
    import matplotlib.pyplot as plt
    from IPython.display import display
    for k in keys:
        fig = figure(k)
        display(fig)
        plt.close(fig)


# ---- Teks & cek silang ----------------------------------------------------------------------------------
def snippet(func, max_lines: int = 15) -> str:
    """Kode fungsi tanpa docstring, maksimal max_lines baris (sisanya diganti '...')."""
    src = textwrap.dedent(inspect.getsource(func))
    lines = src.splitlines()
    out, in_doc, doc_done = [], False, False
    for ln in lines:
        s = ln.strip()
        if not doc_done and (s.startswith('"""') or in_doc):
            if s.startswith('"""') and not in_doc:
                in_doc = not (s.count('"""') >= 2)
                doc_done = not in_doc
            elif '"""' in s:
                in_doc, doc_done = False, True
            continue
        if s:                                   # baris kosong dibuang supaya potongan tetap ringkas
            out.append(ln)
    if len(out) > max_lines:
        out = out[:max_lines - 1] + ["    ..."]
    return "\n".join(out)


def fmt_id(x: float, nd: int = 2, sign: bool = False, pct: bool = False) -> str:
    """Angka gaya Indonesia: koma desimal, minus U+2212, opsional tanda +, opsional persen."""
    v = x * 100 if pct else x
    s = f"{abs(v):,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    if v < 0:
        s = "−" + s
    elif sign:
        s = "+" + s
    return s + ("%" if pct else "")


def crosscheck(items: list[tuple[str, str]], text: str) -> pd.DataFrame:
    """items = (label, string angka yang diharapkan muncul). Cek apakah tiap string ada di teks (minus ASCII juga diterima)."""
    norm = text.replace("-", "−")
    rows = [{"angka": label, "nilai_dari_reports": s, "ada_di_README": s.replace("-", "−") in norm} for label, s in items]
    return pd.DataFrame(rows)
