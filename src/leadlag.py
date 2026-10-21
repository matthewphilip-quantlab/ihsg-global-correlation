"""Lead-lag, presedensi prediktif (Granger), dan sinyal semalam EIDO (Phase 4).

Semua estimasi memakai data in-sample saja (s/d akhir screening window, 2019-12-31). Hasilnya
"presedensi/asosiasi", bukan sinyal: klaim sinyal hanya lewat OOS 2020+ di Phase 6.

Isi modul:
- in_sample / drop_flagged                      : kebersihan OOS dan pembuangan kuotasi basi
- cross_correlation / ccf_peaks                 : IHSG(t+k) vs seri(t), k = -K..+K, per pasangan tanggal sama-sama buka
- grid_returns / lead_frame / pair_lead_betas   : regresi lead IHSG(t+1) ~ seri(t), HAC
- granger_pair / granger_table                  : presedensi prediktif dua arah, lag via AIC/BIC, Wald robust
- eido_frame / cointegration / half_life / nested_regressions : premium EIDO vs IHSG dalam USD
- open_gap_split / gap_lead_corr                : sensitivitas gap pembukaan (EKSPLORATIF)
- phase4_summary
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from . import analysis as an
from .quality import to_returns


# ---- Kebersihan data ----------------------------------------------------------------------
def in_sample(df: pd.DataFrame | pd.Series, end: str) -> pd.DataFrame | pd.Series:
    """Potong ke tanggal <= end. Dipanggil sebelum estimasi apa pun supaya data 2020+ tidak tersentuh."""
    out = df.loc[:end]
    if len(out) and out.index.max() > pd.Timestamp(end):
        raise ValueError("data melewati batas in-sample")
    return out


def drop_flagged(close: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Harga yang ditandai (mis. kuotasi basi) dijadikan NaN = dianggap tidak ada transaksi."""
    out = close.copy()
    for col in flags.columns:
        if col in out:
            f = flags[col].reindex(out.index).fillna(False).astype(bool)
            out.loc[f, col] = np.nan
    return out


# ---- Cross-correlation --------------------------------------------------------------------
def cross_correlation(daily: pd.DataFrame, ids: list[str], max_k: int) -> pd.DataFrame:
    """Korelasi IHSG(t+k) vs seri(t), dengan t = urutan tanggal yang sama-sama buka untuk pasangan itu.
    k > 0: seri mendahului IHSG. p_bh dikoreksi Benjamini-Hochberg atas seluruh tabel."""
    rows = []
    for sid in ids:
        d = daily[daily["id"] == sid].sort_values("Date").reset_index(drop=True)
        for k in range(-max_k, max_k + 1):
            rows.append({"id": sid, "k": k, **an.corr_with_ci(d["r_x"], d["r_target"].shift(-k))})
    out = pd.DataFrame(rows).assign(_g=1)
    return an.add_bh(out, "p", ["_g"]).drop(columns="_g")


def ccf_peaks(ccf: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    """Lag dengan |r| terbesar per seri, plus r di k = 0 dan k = +1, dan sesi perdagangannya."""
    peak = ccf.loc[ccf.groupby("id")["r"].apply(lambda s: s.abs().idxmax())].set_index("id")
    r = ccf.pivot(index="id", columns="k", values="r")
    out = pd.DataFrame({"k_puncak": peak["k"], "r_puncak": peak["r"], "p_bh_puncak": peak["p_bh"],
                        "r_k0": r[0], "r_k1": r[1]})
    return out.join(series.set_index("id")[["name", "session"]]).sort_values(["session", "k_puncak"])


# ---- Regresi lead ---------------------------------------------------------------------------
def grid_returns(close: pd.DataFrame, cols: list[str], transforms: dict[str, str]) -> pd.DataFrame:
    """Return di tanggal yang SEMUA kolom punya harga (tanpa forward-fill)."""
    df = close[list(cols)].dropna()
    return pd.DataFrame({c: to_returns(df[c], transforms.get(c, "log")) for c in cols}).iloc[1:]


def lead_frame(ret: pd.DataFrame, target_col: str, lead_col: str = "target_lead") -> pd.DataFrame:
    """Tambah kolom target di tanggal grid berikutnya (t+1); baris terakhir dibuang."""
    out = ret.assign(**{lead_col: ret[target_col].shift(-1)})
    return out.iloc[:-1]


def pair_lead_betas(daily: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    """Regresi satu faktor IHSG(t+1) ~ seri(t) per pasangan (tanggal sama-sama buka), SE HAC."""
    rows = []
    for sid in ids:
        d = daily[daily["id"] == sid].sort_values("Date")
        f = pd.DataFrame({"y": d["r_target"].shift(-1).to_numpy(), sid: d["r_x"].to_numpy()}).dropna()
        coef, model = an.factor_regression(f, "y", [sid], None, None)
        rows.append({"id": sid, "beta": coef.loc[sid, "beta"], "ci_lo": coef.loc[sid, "ci_lo"],
                     "ci_hi": coef.loc[sid, "ci_hi"], "p": coef.loc[sid, "p"], "r2": model["r2"],
                     "n": model["n_minggu"], "hac_lags": model["hac_lags"]})
    out = pd.DataFrame(rows).assign(_g=1)
    return an.add_bh(out, "p", ["_g"]).drop(columns="_g").set_index("id")


# ---- Granger (presedensi prediktif) ------------------------------------------------------------
def _lagged(df: pd.DataFrame, p: int) -> pd.DataFrame:
    return pd.concat({f"{c}_L{lag}": df[c].shift(lag) for c in df.columns for lag in range(1, p + 1)}, axis=1)


def granger_pair(df: pd.DataFrame, maxlags: int, ic: str = "bic") -> dict:
    """Uji presedensi prediktif dua arah untuk dua kolom [target, seri].
    Orde lag dipilih dari VAR (AIC & BIC); yang dipakai = `ic` (minimal 1). Tiap arah:
    OLS efek ~ lag efek + lag penyebab, SE HAC, lalu uji Wald F bahwa semua lag penyebab = 0."""
    df = df.dropna()
    sel = VAR(df.to_numpy()).select_order(maxlags).selected_orders
    p = max(1, int(sel[ic]))
    lags = _lagged(df, p)
    out = {"lag_aic": int(sel["aic"]), "lag_bic": int(sel["bic"]), "lag_dipakai": p}
    a, b = df.columns
    for cause, effect, key in ((b, a, "seri_ke_target"), (a, b, "target_ke_seri")):
        X = sm.add_constant(lags)
        data = pd.concat([df[effect].rename("_y"), X], axis=1).dropna()
        fit = sm.OLS(data["_y"], data.drop(columns="_y")).fit(cov_type="HAC",
                                                               cov_kwds={"maxlags": an.hac_lags(len(data))})
        names = [f"{cause}_L{lag}" for lag in range(1, p + 1)]
        R = np.zeros((p, len(fit.params)))
        for i, nm in enumerate(names):
            R[i, list(fit.params.index).index(nm)] = 1
        w = fit.wald_test(R, use_f=True, scalar=True)
        out[f"F_{key}"] = float(w.statistic)
        out[f"p_{key}"] = float(w.pvalue)
        out[f"n_{key}"] = int(fit.nobs)
    return out


def granger_table(daily: pd.DataFrame, ids: list[str], maxlags: int, ic: str = "bic") -> pd.DataFrame:
    rows = []
    for sid in ids:
        d = daily[daily["id"] == sid].sort_values("Date")
        rows.append({"id": sid, **granger_pair(d[["r_target", "r_x"]], maxlags, ic)})
    out = pd.DataFrame(rows).assign(_g=1)
    out = an.add_bh(out, "p_seri_ke_target", ["_g"])
    return an.add_bh(out, "p_target_ke_seri", ["_g"]).drop(columns="_g").set_index("id")


# ---- EIDO: premium, kointegrasi, asosiasi semalam -------------------------------------------------
def eido_frame(close: pd.DataFrame, target_id: str, fx_id: str, eido_id: str, extra_ids: list[str],
               transforms: dict[str, str]) -> pd.DataFrame:
    """Grid tanggal yang semua seri buka (IHSG, USD/IDR, EIDO, seri pembanding).

    Timing pada tanggal t: IHSG tutup 16:00 WIB; IDR=X dan EIDO tutup di sesi AS (dini hari WIB t+1).
    IHSG dalam USD = IHSG(t) / USDIDR(t), sehingga premium = log EIDO(t) - log IHSG_USD(t) memuat pergerakan
    saham Indonesia (dalam USD) SETELAH IHSG tutup. Target = return IHSG di tanggal grid berikutnya."""
    cols = list(dict.fromkeys([target_id, fx_id, eido_id] + list(extra_ids)))
    lv = close[cols].dropna()
    f = pd.DataFrame({"log_eido": np.log(lv[eido_id]), "log_target_usd": np.log(lv[target_id] / lv[fx_id])})
    f["premium"] = f["log_eido"] - f["log_target_usd"]
    f["r_eido"] = f["log_eido"].diff()
    f["r_target_usd"] = f["log_target_usd"].diff()
    f["d_premium"] = f["premium"].diff()          # = r_eido - r_target_usd: bagian yang belum ada di close IHSG
    f["r_target"] = to_returns(lv[target_id], "log")
    for c in extra_ids:
        f[f"r_{c}"] = to_returns(lv[c], transforms.get(c, "log"))
    f["target_lead"] = f["r_target"].shift(-1)
    return f.iloc[1:-1]


def half_life(spread: pd.Series) -> float:
    """Half-life (hari bursa) dari AR(1): d s_t = lambda s_(t-1) + e  ->  HL = -ln 2 / ln(1 + lambda)."""
    s = spread.dropna()
    ds, lag = s.diff().iloc[1:], s.shift(1).iloc[1:]
    lam = sm.OLS(ds, sm.add_constant(lag)).fit().params.iloc[1]
    return float(-math.log(2) / math.log(1 + lam)) if -1 < lam < 0 else np.inf


def cointegration(log_y: pd.Series, log_x: pd.Series, alpha: float = 0.05) -> dict:
    """Engle-Granger (konstanta, dan konstanta+tren sebagai cek), Johansen (cek), ADF premium mentah [1, -1].
    Kalau Engle-Granger (konstanta) signifikan: spread = residual regresi kointegrasi, z-score, half-life."""
    df = pd.concat([log_y, log_x], axis=1).dropna()
    y, x = df.iloc[:, 0], df.iloc[:, 1]
    eg_c, eg_ct = coint(y, x, trend="c"), coint(y, x, trend="ct")
    joh = coint_johansen(df.to_numpy(), det_order=0, k_ar_diff=1)
    adf_raw = adfuller(y - x, regression="c", autolag="AIC", result_object=False)
    ols = sm.OLS(y, sm.add_constant(x)).fit()
    spread = ols.resid
    trend = pd.Series(np.arange(len(df), dtype=float), index=df.index, name="tren")
    spread_ct = sm.OLS(y, sm.add_constant(pd.concat([x, trend], axis=1))).fit().resid
    res = {"n": int(len(df)), "mulai": df.index.min(), "akhir": df.index.max(),
           "eg_c_stat": float(eg_c[0]), "eg_c_p": float(eg_c[1]),
           "eg_ct_stat": float(eg_ct[0]), "eg_ct_p": float(eg_ct[1]),
           "johansen_trace_r0": float(joh.lr1[0]), "johansen_crit95_r0": float(joh.cvt[0, 1]),
           "johansen_r0_ditolak": bool(joh.lr1[0] > joh.cvt[0, 1]),
           "adf_premium_mentah_stat": float(adf_raw[0]), "adf_premium_mentah_p": float(adf_raw[1]),
           "alpha_kointegrasi": float(ols.params.iloc[0]), "beta_kointegrasi": float(ols.params.iloc[1]),
           "half_life_hari_ct_info": half_life(spread_ct),   # informasi saja; keputusan memakai Engle-Granger konstanta
           "terkointegrasi": bool(eg_c[1] < alpha)}
    if res["terkointegrasi"]:
        res["spread_mean"], res["spread_std"] = float(spread.mean()), float(spread.std())
        res["half_life_hari"] = half_life(spread)
    return res


def spread_zscore(log_y: pd.Series, log_x: pd.Series, coint_res: dict) -> pd.Series:
    """z-score spread dengan parameter in-sample dari `cointegration`."""
    s = log_y - (coint_res["alpha_kointegrasi"] + coint_res["beta_kointegrasi"] * log_x)
    return ((s - coint_res["spread_mean"]) / coint_res["spread_std"]).rename("spread_z")


def nested_regressions(frame: pd.DataFrame, y: str, models: dict[str, list[str]], base: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Beberapa regresi OLS-HAC pada sampel yang sama. Untuk model yang memuat semua regresor `base`,
    dilaporkan tambahan R² dan uji Wald F (HAC) bahwa regresor tambahan = 0."""
    cols = sorted({c for v in models.values() for c in v})
    data = frame[[y] + cols].dropna()
    lags = an.hac_lags(len(data))
    fits = {m: sm.OLS(data[y], sm.add_constant(data[v])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
            for m, v in models.items()}
    coef_rows, model_rows = [], []
    for m, fit in fits.items():
        ci = fit.conf_int()
        for c in models[m]:
            coef_rows.append({"model": m, "variabel": c, "beta": fit.params[c], "ci_lo": ci.loc[c, 0],
                              "ci_hi": ci.loc[c, 1], "t_hac": fit.tvalues[c], "p": fit.pvalues[c]})
        row = {"model": m, "regresor": " + ".join(models[m]), "n": int(fit.nobs), "hac_lags": lags,
               "r2": fit.rsquared, "adj_r2": fit.rsquared_adj}
        extra = [c for c in models[m] if c not in models[base]]
        if m != base and set(models[base]) <= set(models[m]) and extra:
            R = np.zeros((len(extra), len(fit.params)))
            for i, c in enumerate(extra):
                R[i, list(fit.params.index).index(c)] = 1
            w = fit.wald_test(R, use_f=True, scalar=True)
            row.update({"tambahan_r2_vs_base": fit.rsquared - fits[base].rsquared,
                        "F_tambahan": float(w.statistic), "p_tambahan": float(w.pvalue)})
        model_rows.append(row)
    return pd.DataFrame(coef_rows), pd.DataFrame(model_rows).set_index("model")


# ---- Gap pembukaan (EKSPLORATIF) -------------------------------------------------------------------
def open_gap_split(ohlc: pd.DataFrame, rel_tol: float = 1e-6) -> pd.DataFrame:
    """gap = log(Open_t / Close_(t-1)), intraday = log(Close_t / Open_t). Hari dengan Open = Close kemarin
    (Open diisi data penutupan) dibuang, sesuai rencana sensitivitas EKSPLORATIF."""
    c, o = ohlc["Close"], ohlc["Open"]
    prev = c.shift(1)
    bad = ((o - prev).abs() <= rel_tol * prev.abs()) | o.isna() | (o <= 0)
    out = pd.DataFrame({"gap": np.log(o / prev), "intraday": np.log(c / o)})
    out[bad] = np.nan
    return out.iloc[1:]


def gap_lead_corr(daily: pd.DataFrame, split: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    """Korelasi seri(t) dengan gap pembukaan dan intraday IHSG di tanggal pasangan berikutnya."""
    rows = []
    for sid in ids:
        d = daily[daily["id"] == sid].sort_values("Date")
        nxt = split.reindex(d["Date"]).shift(-1)
        for part in ("gap", "intraday"):
            rows.append({"id": sid, "bagian": part,
                         **an.corr_with_ci(d["r_x"].reset_index(drop=True), nxt[part].reset_index(drop=True))})
    out = pd.DataFrame(rows).assign(_g=1)
    return an.add_bh(out, "p", ["_g"]).drop(columns="_g")


# ---- Ringkasan & output --------------------------------------------------------------------------
def phase4_summary(peaks: pd.DataFrame, lead_multi: pd.DataFrame, lead_model: dict, granger: pd.DataFrame,
                   coint_res: dict, eido_models: pd.DataFrame, names: dict, fx_id: str,
                   lead_1f: pd.DataFrame | None = None, alpha: float = 0.05) -> str:
    lines = ["### Ringkasan Phase 4 (presedensi in-sample 2010-2019, BUKAN sinyal)", ""]
    after = peaks[peaks["session"] == "after"]
    sync = peaks[peaks["session"] == "sync"]
    by_k = after.groupby("k_puncak").apply(lambda d: ", ".join(names[i] for i in d.index), include_groups=False)
    lines.append(f"- **Pola waktu:** {int((sync['k_puncak'] == 0).sum())} dari {len(sync)} seri \"sync\" puncaknya di k = 0. "
                 f"Seri \"after\": " + "; ".join(f"k = {int(k):+d}: {v}" for k, v in by_k.items()) + ".")
    sig = lead_multi[lead_multi["p_bh"] < alpha]
    parts = []
    for i, r in sig.iterrows():
        txt = f"{names[i]} +1% → IHSG besok {r.beta * 100:+.0f} bp"
        if lead_1f is not None and i in lead_1f.index and np.sign(lead_1f.loc[i, "beta"]) != np.sign(r.beta):
            txt += f" (tanda berbalik dari satu faktor {lead_1f.loc[i, 'beta'] * 100:+.0f} bp: efek tumpang tindih)"
        parts.append(txt)
    lines.append(f"- **Malam ini di New York → IHSG besok (regresi lead multi-faktor, R² {lead_model['r2']:.3f}):** "
                 + ("; ".join(parts) if parts else "tidak ada seri yang signifikan setelah koreksi BH") + ".")
    g = granger[granger["p_seri_ke_target_bh"] < alpha]
    lines.append(f"- **Presedensi prediktif (Granger, BH 5%):** seri → IHSG pada "
                 + (", ".join(names[i] for i in g.index) if len(g) else "tidak ada") + "; IHSG → seri pada "
                 + (", ".join(names[i] for i in granger[granger["p_target_ke_seri_bh"] < alpha].index) or "tidak ada") + ".")
    if fx_id in peaks.index:
        r = peaks.loc[fx_id]
        lines.append(f"- **Rupiah (IDR=X):** r hari yang sama {r.r_k0:+.3f}, hari berikutnya {r.r_k1:+.3f}, puncak di k = {int(r.k_puncak)}.")
    co = ("terkointegrasi" if coint_res["terkointegrasi"] else "TIDAK terkointegrasi") + \
         f" menurut aturan utama (Engle-Granger konstanta p = {coint_res['eg_c_p']:.3f}); cek: Engle-Granger + tren " \
         f"p = {coint_res['eg_ct_p']:.3g}, Johansen r=0 {'ditolak' if coint_res['johansen_r0_ditolak'] else 'tidak ditolak'}"
    hl = f", half-life spread {coint_res['half_life_hari']:.1f} hari bursa" if coint_res.get("half_life_hari") else ""
    lines.append(f"- **EIDO vs IHSG dalam USD:** {co}{hl}.")
    for m, r in eido_models.dropna(subset=["p_tambahan"]).iterrows():
        lines.append(f"  - {m}: tambahan R² {r.tambahan_r2_vs_base:+.3f} di atas model pembanding, p (Wald HAC) = {r.p_tambahan:.3g}.")
    return "\n".join(lines)

