"""Korelasi statis, regresi faktor, dan korelasi asimetris (Phase 3). Semua hasil DESKRIPTIF (asosiasi in-sample).

Isi modul:
- period_bounds / fisher_ci / fisher_diff / corr_with_ci : dasar statistik korelasi
- correlation_table / compare_targets / session_gap_table : korelasi per periode, frekuensi, metode
- hac_lags / factor_regression / vif_table / single_factor_betas / scenario_text : beta multifaktor (HAC)
- asymmetric_table / expected_conditional_corr / stationary_bootstrap_indices : korelasi pasar turun vs naik
- corr_matrix / subperiod_test / stale_sensitivity : matriks, uji ulang hipotesis lama, sensitivitas flag
- phase3_summary / save_tables : ringkasan dan file hasil
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .quality import _json_default

SPEARMAN_SE_FACTOR = 1.06   # Fieller-Hartley-Pearson: Var(atanh r_s) ~ 1.06 / (n - 3)
METHODS = ("pearson", "spearman")


# ---- Dasar statistik ------------------------------------------------------------------
def period_bounds(cfg: dict, analysis_start: str) -> dict[str, tuple[str, str | None]]:
    """Tiga periode: penuh, screening (dikunci), pasca-screening."""
    win = cfg["project"]["screening_window"]
    return {"penuh": (analysis_start, None),
            "screening": (win[0], win[1]),
            "pasca": (cfg["analysis"]["post_screening_start"], None)}


def fisher_ci(r: float, n: int, alpha: float = 0.05, se_factor: float = 1.0) -> tuple[float, float]:
    if n is None or n <= 3 or not np.isfinite(r):
        return np.nan, np.nan
    z, se = np.arctanh(np.clip(r, -0.999999, 0.999999)), math.sqrt(se_factor / (n - 3))
    q = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - q * se)), float(np.tanh(z + q * se))


def fisher_diff(r1: float, n1: int, r2: float, n2: int, se_factor: float = 1.0) -> tuple[float, float]:
    """Uji selisih dua korelasi dari sampel INDEPENDEN (Fisher z). Return (z, p dua sisi)."""
    if min(n1, n2) <= 3 or not (np.isfinite(r1) and np.isfinite(r2)):
        return np.nan, np.nan
    z = (np.arctanh(r1) - np.arctanh(r2)) / math.sqrt(se_factor / (n1 - 3) + se_factor / (n2 - 3))
    return float(z), float(2 * stats.norm.sf(abs(z)))


def corr_with_ci(x: pd.Series, y: pd.Series, method: str = "pearson") -> dict:
    pair = pd.concat([x, y], axis=1).dropna()
    n = len(pair)
    if n < 10:
        return {"r": np.nan, "n": n, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan}
    a, b = pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy()
    res = stats.pearsonr(a, b) if method == "pearson" else stats.spearmanr(a, b)
    r, p = float(res[0]), float(res[1])
    lo, hi = fisher_ci(r, n, se_factor=1.0 if method == "pearson" else SPEARMAN_SE_FACTOR)
    return {"r": r, "n": n, "ci_lo": lo, "ci_hi": hi, "p": p}


def in_period(dates: pd.Series, start, end) -> pd.Series:
    """Mask tanggal di [start, end]; end None = sampai akhir data."""
    m = dates >= pd.Timestamp(start)
    return m if end is None else m & (dates <= pd.Timestamp(end))


def add_bh(df: pd.DataFrame, p_col: str, by: list[str], out_col: str | None = None) -> pd.DataFrame:
    """p-value terkoreksi Benjamini-Hochberg (FDR) per kelompok tabel."""
    out_col = out_col or f"{p_col}_bh"
    df = df.copy()
    df[out_col] = np.nan
    for _, g in df.groupby(by):
        p = g[p_col]
        ok = p.notna()
        if ok.any():
            df.loc[p[ok].index, out_col] = multipletests(p[ok], method="fdr_bh")[1]
    return df


# ---- Korelasi per periode / frekuensi / metode ------------------------------------------
def correlation_table(daily: pd.DataFrame, weekly: pd.DataFrame, target_col: str, ids: list[str],
                      periods: dict[str, tuple]) -> pd.DataFrame:
    """Korelasi tiap seri vs target. Harian = return hari yang sama dari panel pasangan (tanggal sama-sama buka);
    mingguan = W-FRI. Kolom p_bh dikoreksi per (periode, frekuensi, metode)."""
    rows = []
    for pname, (start, end) in periods.items():
        d = daily[in_period(daily["Date"], start, end)]
        w = weekly.loc[start:end]
        for sid in ids:
            dd = d[d["id"] == sid]
            for method in METHODS:
                rows.append({"periode": pname, "frekuensi": "harian", "metode": method, "id": sid,
                             **corr_with_ci(dd["r_x"], dd["r_target"], method)})
                rows.append({"periode": pname, "frekuensi": "mingguan", "metode": method, "id": sid,
                             **corr_with_ci(w[sid], w[target_col], method)})
    return add_bh(pd.DataFrame(rows), "p", ["periode", "frekuensi", "metode"])


def compare_targets(tab_a: pd.DataFrame, tab_b: pd.DataFrame, label_a: str, label_b: str) -> pd.DataFrame:
    """Korelasi yang sama terhadap dua versi target (mis. IHSG dalam IDR vs USD). Sampelnya tumpang tindih,
    jadi selisihnya dilaporkan tanpa uji Fisher (uji itu mengasumsikan sampel independen)."""
    keys = ["periode", "frekuensi", "metode", "id"]
    m = tab_a[keys + ["r", "n"]].merge(tab_b[keys + ["r", "n"]], on=keys, suffixes=(f"_{label_a}", f"_{label_b}"))
    m["selisih"] = m[f"r_{label_b}"] - m[f"r_{label_a}"]
    return m


def session_gap_table(tab: pd.DataFrame, series: pd.DataFrame, period: str, method: str = "pearson") -> pd.DataFrame:
    """Korelasi harian (hari yang sama) vs mingguan per seri, dengan sesi perdagangannya.
    Seri sesi 'after' tutup setelah IHSG, jadi sebagian reaksinya baru muncul di IHSG hari berikutnya."""
    t = tab[(tab["periode"] == period) & (tab["metode"] == method)]
    p = t.pivot(index="id", columns="frekuensi", values="r")
    out = p.join(series.set_index("id")[["name", "session"]], how="left")
    out["rasio_harian_mingguan"] = out["harian"] / out["mingguan"]
    return out[["name", "session", "harian", "mingguan", "rasio_harian_mingguan"]].sort_values(["session", "mingguan"])


# ---- Regresi faktor mingguan (HAC) -------------------------------------------------------
def hac_lags(n: int) -> int:
    """Aturan Newey-West yang dikunci: floor(4 (T/100)^(2/9))."""
    return int(math.floor(4 * (n / 100) ** (2 / 9)))


def _design(weekly: pd.DataFrame, target_col: str, ids: list[str], start, end) -> pd.DataFrame:
    return weekly.loc[start:end, [target_col] + list(ids)].dropna()


def factor_regression(weekly: pd.DataFrame, target_col: str, ids: list[str], start, end) -> tuple[pd.DataFrame, dict]:
    """OLS target ~ semua seri sekaligus, SE Newey-West. Return (tabel koefisien, statistik model)."""
    df = _design(weekly, target_col, ids, start, end)
    lags = hac_lags(len(df))
    fit = sm.OLS(df[target_col], sm.add_constant(df[list(ids)])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    ci = fit.conf_int()
    coef = pd.DataFrame({"beta": fit.params, "se_hac": fit.bse, "t_hac": fit.tvalues, "p": fit.pvalues,
                         "ci_lo": ci[0], "ci_hi": ci[1]}).drop(index="const")
    coef.index.name = "id"
    coef = add_bh(coef.reset_index().assign(_g=1), "p", ["_g"]).drop(columns="_g").set_index("id")
    model = {"n_minggu": int(fit.nobs), "hac_lags": lags, "r2": float(fit.rsquared), "adj_r2": float(fit.rsquared_adj),
             "mulai": df.index.min(), "akhir": df.index.max()}
    return coef, model


def vif_table(weekly: pd.DataFrame, ids: list[str], start, end) -> pd.Series:
    X = sm.add_constant(weekly.loc[start:end, list(ids)].dropna()).to_numpy()
    return pd.Series([variance_inflation_factor(X, i + 1) for i in range(len(ids))], index=list(ids), name="vif")


def single_factor_betas(weekly: pd.DataFrame, target_col: str, ids: list[str], start, end) -> pd.DataFrame:
    rows = []
    for sid in ids:
        coef, model = factor_regression(weekly, target_col, [sid], start, end)
        rows.append({"id": sid, "beta_1f": coef.loc[sid, "beta"], "ci_lo_1f": coef.loc[sid, "ci_lo"],
                     "ci_hi_1f": coef.loc[sid, "ci_hi"], "p_1f": coef.loc[sid, "p"], "r2_1f": model["r2"]})
    return pd.DataFrame(rows).set_index("id")


def scenario_text(sid: str, name: str, beta: float, ci_lo: float, ci_hi: float, scfg: dict,
                  target_name: str = "IHSG") -> str:
    """Beta dalam bahasa skenario. Return log mingguan, jadi beta ~ elastisitas: guncangan s% -> beta*s%."""
    spec = scfg.get("by_id", {}).get(sid, {})
    shock = spec.get("shock_pct", scfg["default_shock_pct"])
    phrase = spec.get("phrase", f"{name} naik {shock:g}%")
    return (f"{phrase} → {target_name} rata-rata {beta * shock:+.2f}% di minggu yang sama "
            f"(CI 95% {ci_lo * shock:+.2f}% s/d {ci_hi * shock:+.2f}%), seri Core lain dikendalikan")


def scenario_lines(coef: pd.DataFrame, scenarios: dict[str, str], vif_thr: float, alpha: float = 0.05) -> list[str]:
    """Satu kalimat per seri: skenario beta, signifikansi setelah BH, dan peringatan kalau VIF tinggi atau
    tanda beta multifaktor berbalik dari beta satu faktor (tumpang tindih antar-seri, bukan efek terpisah).
    `coef` butuh kolom beta, p_bh, vif, beta_1f, ci_lo_1f, ci_hi_1f, r2_1f, label."""
    lines = []
    for sid, r in coef.iterrows():
        txt = f"- {scenarios[sid]}; {'signifikan' if r.p_bh < alpha else 'tidak signifikan'} setelah koreksi BH"
        if np.sign(r.beta) != np.sign(r.beta_1f) and r.p_bh < alpha:
            txt += (f". **Tanda berbalik** dari regresi satu faktor ({r.beta_1f:+.3f}): ini efek tumpang tindih "
                    "dengan seri lain, bukan berarti seri ini menekan IHSG")
        if r.vif > vif_thr or np.sign(r.beta) != np.sign(r.beta_1f):
            txt += (f" — VIF {r.vif:.1f}; beta satu faktor {r.beta_1f:+.3f} [{r.ci_lo_1f:+.3f}, {r.ci_hi_1f:+.3f}], "
                    f"R² {r.r2_1f:.2f}")
        if r.label:
            txt += f" *({r.label})*"
        lines.append(txt + ".")
    return lines


def usd_effect_text(usd_cmp: pd.DataFrame, fx_id: str, small: float = 0.03) -> str:
    """Rata-rata perubahan besaran korelasi mingguan (Pearson), |r_usd| - |r_idr|, untuk seri non-kurs, per periode.
    Memakai besaran supaya seri berkorelasi negatif (mis. VIX) tidak saling meniadakan dengan yang positif."""
    c = usd_cmp[(usd_cmp["frekuensi"] == "mingguan") & (usd_cmp["metode"] == "pearson") & (usd_cmp["id"] != fx_id)]
    parts = []
    for per, d in c.groupby("periode"):
        m = (d["r_usd"].abs() - d["r_idr"].abs()).mean()
        verdict = "praktis sama" if abs(m) < small else ("lebih kuat dalam USD" if m > 0 else "lebih lemah dalam USD")
        parts.append(f"{per} {m:+.2f} ({verdict})")
    return "; ".join(parts)


# ---- Korelasi asimetris ------------------------------------------------------------------
def _masked_corr(y: np.ndarray, x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Korelasi Pearson y-x di dalam mask m. Bekerja per baris kalau input 2D (baris = replikasi bootstrap)."""
    w = m.astype(float)
    k = w.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        my, mx = (w * y).sum(-1) / k, (w * x).sum(-1) / k
        dy, dx = y - np.expand_dims(my, -1), x - np.expand_dims(mx, -1)
        cov = (w * dy * dx).sum(-1)
        r = cov / np.sqrt((w * dy ** 2).sum(-1) * (w * dx ** 2).sum(-1))
    return np.where(k >= 10, r, np.nan)


def _masks(c: np.ndarray, mode: str, q: float, ref: np.ndarray | None = None):
    """Kelompok 'turun' & 'naik'. mode 'tanda': c<0 vs c>0; 'exceedance': <= persentil q vs >= persentil 1-q
    (persentil dari `ref`, default c sendiri)."""
    if mode == "tanda":
        return c < 0, c > 0
    ref = c if ref is None else ref
    lo, hi = np.quantile(ref, q, axis=-1, keepdims=ref.ndim > 1), np.quantile(ref, 1 - q, axis=-1, keepdims=ref.ndim > 1)
    return c <= lo, c >= hi


def expected_conditional_corr(data: np.ndarray, mode: str, q: float, n_mc: int, seed: int) -> tuple[float, float]:
    """Korelasi kondisional y-x (kolom 0, 1) yang DIHARAPKAN kalau data normal multivariat dengan
    mean & kovarians sampel, dikondisikan pada kolom terakhir (Boyer-Gibson-Loretan, versi simulasi)."""
    rng = np.random.default_rng(seed)
    sim = rng.multivariate_normal(data.mean(0), np.cov(data, rowvar=False), size=n_mc)
    down, up = _masks(sim[:, -1], mode, q)
    return float(_masked_corr(sim[:, 0], sim[:, 1], down)), float(_masked_corr(sim[:, 0], sim[:, 1], up))


def stationary_bootstrap_indices(n: int, n_boot: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Indeks stationary bootstrap (Politis-Romano), shape (n_boot, n). Blok berpanjang geometrik, melingkar."""
    idx = np.empty((n_boot, n), dtype=int)
    idx[:, 0] = rng.integers(n, size=n_boot)
    new_block = rng.random((n_boot, n)) < 1 / mean_block
    starts = rng.integers(n, size=(n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def asymmetric_table(weekly: pd.DataFrame, target_col: str, ids: list[str], cond_ids: list[str],
                     start, end, acfg: dict, modes: tuple[str, ...] = ("tanda", "exceedance")) -> pd.DataFrame:
    """Korelasi target-seri di minggu 'turun' vs 'naik' menurut seri pengkondisi (cond_ids, plus 'sendiri').

    Dilaporkan: r_turun, r_naik, uji Fisher z (naif, menganggap kelompok independen), nilai yang diharapkan
    di bawah normal multivariat, dan 'asimetri berlebih' = (r_turun - r_naik) - (harapan_turun - harapan_naik)
    dengan CI dan p dari stationary block bootstrap. Memotong sampel saja sudah menurunkan korelasi,
    karena itu pembanding normal diperlukan."""
    rng = np.random.default_rng(acfg["seed"])
    q = acfg["exceedance_q"]
    rows = []
    for cond in list(cond_ids) + ["sendiri"]:
        for sid in ids:
            c_col = sid if cond == "sendiri" else cond
            cols = [target_col, sid] + ([c_col] if c_col != sid else [])   # pengkondisi = kolom terakhir
            df = weekly.loc[start:end, cols].dropna()
            data = df.to_numpy()
            y, x, c = data[:, 0], data[:, 1], data[:, -1]
            for mode in modes:
                down, up = _masks(c, mode, q)
                r_d, r_u = float(_masked_corr(y, x, down)), float(_masked_corr(y, x, up))
                n_d, n_u = int(down.sum()), int(up.sum())
                z, p = fisher_diff(r_d, n_d, r_u, n_u)
                e_d, e_u = expected_conditional_corr(data, mode, q, acfg["n_mc"], acfg["seed"])
                idx = stationary_bootstrap_indices(len(df), acfg["n_boot"], acfg["mean_block_weeks"], rng)
                yb, xb, cb = y[idx], x[idx], c[idx]
                bd, bu = _masks(cb, mode, q, ref=cb)
                excess_b = (_masked_corr(yb, xb, bd) - _masked_corr(yb, xb, bu)) - (e_d - e_u)
                excess_b = excess_b[np.isfinite(excess_b)]
                excess = (r_d - r_u) - (e_d - e_u)
                p_boot = float(min(1.0, 2 * min((excess_b <= 0).mean(), (excess_b >= 0).mean())))
                rows.append({"kondisi": cond, "mode": mode, "id": sid, "n_turun": n_d, "n_naik": n_u,
                             "r_turun": r_d, "r_naik": r_u, "selisih": r_d - r_u, "z_fisher": z, "p_fisher": p,
                             "harapan_turun": e_d, "harapan_naik": e_u, "asimetri_berlebih": excess,
                             "ci_lo_boot": float(np.quantile(excess_b, 0.025)),
                             "ci_hi_boot": float(np.quantile(excess_b, 0.975)), "p_boot": p_boot})
    out = pd.DataFrame(rows)
    out = add_bh(out, "p_fisher", ["kondisi", "mode"])
    return add_bh(out, "p_boot", ["kondisi", "mode"])


# ---- Matriks, sub-periode, sensitivitas ---------------------------------------------------
def corr_matrix(weekly: pd.DataFrame, cols: list[str], start, end) -> pd.DataFrame:
    """Korelasi Pearson mingguan berpasangan (pairwise complete)."""
    return weekly.loc[start:end, list(cols)].corr()


def subperiod_test(weekly_by_variant: dict[str, pd.DataFrame], target_col: str, x_id: str,
                   subperiods: list[list[str]]) -> pd.DataFrame:
    """Korelasi mingguan di dua sub-periode per varian data, plus uji Fisher z selisihnya
    (sub-periode tidak tumpang tindih, jadi asumsi independensi masuk akal)."""
    (a0, a1), (b0, b1) = subperiods
    rows = []
    for name, w in weekly_by_variant.items():
        ra = corr_with_ci(w.loc[a0:a1, x_id], w.loc[a0:a1, target_col])
        rb = corr_with_ci(w.loc[b0:b1, x_id], w.loc[b0:b1, target_col])
        z, p = fisher_diff(ra["r"], ra["n"], rb["r"], rb["n"])
        rows.append({"varian": name, "r_awal": ra["r"], "ci_awal": (ra["ci_lo"], ra["ci_hi"]), "n_awal": ra["n"],
                     "r_akhir": rb["r"], "ci_akhir": (rb["ci_lo"], rb["ci_hi"]), "n_akhir": rb["n"],
                     "selisih": rb["r"] - ra["r"], "z_fisher": z, "p_fisher": p})
    out = pd.DataFrame(rows)
    out.attrs["subperiode"] = f"{a0[:4]}-{a1[:4]} vs {b0[:4]}-{b1[:4]}"
    return out


def stale_sensitivity(daily: pd.DataFrame, x_id: str, periods: dict[str, tuple]) -> pd.DataFrame:
    """Korelasi harian seri vs target, dengan dan tanpa return yang ditandai flag_stale_quote."""
    rows = []
    d = daily[daily["id"] == x_id]
    for pname, (start, end) in periods.items():
        dd = d[in_period(d["Date"], start, end)]
        for label, sub in (("semua return", dd), ("tanpa flag_stale_quote", dd[~dd["flag_stale_quote"]])):
            for method in METHODS:
                rows.append({"periode": pname, "sampel": label, "metode": method, "n_flag": int(dd["flag_stale_quote"].sum()),
                             **corr_with_ci(sub["r_x"], sub["r_target"], method)})
    return pd.DataFrame(rows)


# ---- Ringkasan & output -------------------------------------------------------------------
def rank_stability(tab: pd.DataFrame, freq: str = "mingguan", method: str = "pearson") -> pd.DataFrame:
    """Peringkat |r| per periode dan korelasi peringkat (Kendall tau) antar-periode."""
    t = tab[(tab["frekuensi"] == freq) & (tab["metode"] == method)]
    ranks = t.assign(abs_r=t["r"].abs()).pivot(index="id", columns="periode", values="abs_r").rank(ascending=False)
    return ranks


def phase3_summary(tab: pd.DataFrame, ranks: pd.DataFrame, coef: pd.DataFrame, model: dict, scenarios: dict,
                   usd_cmp: pd.DataFrame, asym: pd.DataFrame, sub: pd.DataFrame, cfg: dict, names: dict) -> str:
    fx_id = cfg["usdidr_retest"]["id"]
    main = tab[(tab["periode"] == "screening") & (tab["frekuensi"] == "mingguan") & (tab["metode"] == "pearson")]
    top = main.assign(a=main["r"].abs()).sort_values("a", ascending=False)
    lines = ["### Ringkasan Phase 3 (deskriptif, asosiasi in-sample)", ""]
    lines.append("- **Seri paling kuat** (korelasi mingguan 2010-2019): "
                 + ", ".join(f"{names[r.id]} {r.r:+.2f}" for r in top.head(3).itertuples()) + ".")
    tau = {f"{a} vs {b}": stats.kendalltau(ranks[a], ranks[b])[0]
           for a, b in (("screening", "pasca"), ("screening", "penuh")) if a in ranks and b in ranks}
    lines.append("- **Stabilitas urutan:** korelasi peringkat (Kendall tau) "
                 + ", ".join(f"{k} = {v:.2f}" for k, v in tau.items())
                 + ". Tiga teratas pasca-2020: "
                 + ", ".join(names[i] for i in ranks["pasca"].sort_values().index[:3]) + ".")
    if fx_id in coef.index:
        lines.append(f"- **Efek kurs:** {scenarios[fx_id]} (R² model 2010-2019 = {model['r2']:.2f}).")
    lines.append("- **IHSG dalam USD vs rupiah** (rata-rata |r_usd| − |r_idr| mingguan, seri non-kurs): "
                 + usd_effect_text(usd_cmp, fx_id) + ".")
    a = asym[(asym["mode"] == "tanda") & (asym["kondisi"] == asym["id"])]
    for r in a.itertuples():
        lines.append(f"- **Pasar turun vs naik ({names[r.id]}):** korelasi IHSG {r.r_turun:+.2f} di minggu {names[r.id]} turun "
                     f"vs {r.r_naik:+.2f} saat naik; setelah dibandingkan dengan yang diharapkan dari distribusi normal, "
                     f"asimetri berlebih {r.asimetri_berlebih:+.2f} (CI bootstrap {r.ci_lo_boot:+.2f} s/d {r.ci_hi_boot:+.2f}; "
                     f"{'CI tidak memuat nol' if r.ci_lo_boot > 0 or r.ci_hi_boot < 0 else 'CI memuat nol, belum meyakinkan'}).")
    clean = sub[sub["varian"] == "bersih (Amandemen 1)"].iloc[0]
    raw = sub.iloc[0]
    lines.append(f"- **Hipotesis lama USD/IDR menguat ({sub.attrs.get('subperiode', '')}):** data mentah {raw.r_awal:+.2f} → "
                 f"{raw.r_akhir:+.2f} (p = {raw.p_fisher:.3f}); data bersih {clean.r_awal:+.2f} → {clean.r_akhir:+.2f} "
                 f"(p = {clean.p_fisher:.2f}). "
                 + ("Pola tidak lagi signifikan dengan data bersih." if clean.p_fisher >= 0.05 else "Pola tetap signifikan."))
    return "\n".join(lines)


def save_tables(out_dir: Path, tables: dict[str, pd.DataFrame], summary: dict) -> list[Path]:
    """Simpan tiap tabel sebagai CSV plus summary.json (dipakai Phase 3 dan seterusnya)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in tables.items():
        p = out_dir / f"{name}.csv"
        df.round(4).to_csv(p, index=not isinstance(df.index, pd.RangeIndex))
        paths.append(p)
    p = out_dir / "summary.json"
    p.write_text(json.dumps({"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), **summary},
                            indent=2, default=_json_default, ensure_ascii=False), encoding="utf-8")
    return paths + [p]
