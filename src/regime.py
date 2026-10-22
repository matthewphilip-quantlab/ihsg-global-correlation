"""Regime & krisis (Phase 5). DESKRIPTIF: boleh memakai seluruh periode, tetapi tidak boleh mengubah
Spesifikasi Phase 6 (README).

Isi modul:
- rolling_corr / rolling_shift_summary            : korelasi bergulir mingguan dan ringkasan pergeseran pasca-break
- episode_window / episode_metrics / episode_corr : event study di jendela [-20, +60] hari bursa target
- event_study / summarize_episodes / event_paths  : tabel per episode x seri, median lintas episode, jalur harga
- crisis_week_mask / crisis_weekly_corr / episode_returns : aset pelindung saat krisis
- quantile_regression / spike_interaction         : hipotesis Brent (berlabel)
- phase5_summary
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

from . import analysis as an


# ---- Rolling correlation -----------------------------------------------------------------------
def rolling_corr(weekly: pd.DataFrame, target_col: str, ids: list[str], window: int, min_frac: float = 0.8) -> pd.DataFrame:
    """Korelasi Pearson bergulir target vs tiap seri; butuh >= min_frac x window minggu lengkap."""
    minp = int(np.ceil(window * min_frac))
    return pd.DataFrame({sid: weekly[target_col].rolling(window, min_periods=minp).corr(weekly[sid]) for sid in ids})


def rolling_shift_summary(roll: pd.DataFrame, break_date: str, split_years: int = 3,
                          pre_start: str | None = None) -> pd.DataFrame:
    """Apakah korelasi pasca-break bergeser sebagai tren (bertahan) atau lonjakan (hanya sesaat)?
    Rata-rata sebelum break, di `split_years` tahun pertama sesudahnya, dan sesudah itu; plus porsi minggu
    pasca-break di luar rentang persentil 10-90 nilai sebelum break, dan nilai terakhir."""
    b = pd.Timestamp(break_date)
    mid = b + pd.DateOffset(years=split_years)
    pre = roll.loc[pre_start:b - pd.Timedelta(days=1)]
    early, late = roll.loc[b:mid - pd.Timedelta(days=1)], roll.loc[mid:]
    post = roll.loc[b:]
    rows = []
    for sid in roll.columns:
        p10, p90 = pre[sid].quantile(0.1), pre[sid].quantile(0.9)
        po = post[sid].dropna()
        rows.append({"id": sid, "rata_pra": pre[sid].mean(), "rata_pasca_awal": early[sid].mean(),
                     "rata_pasca_akhir": late[sid].mean(), "terakhir": roll[sid].dropna().iloc[-1],
                     "porsi_pasca_di_bawah_p10_pra": (po < p10).mean(), "porsi_pasca_di_atas_p90_pra": (po > p90).mean()})
    out = pd.DataFrame(rows).set_index("id")
    out.attrs["periode"] = (f"pra < {b:%Y-%m-%d}; pasca awal {b:%Y}-{(mid - pd.Timedelta(days=1)):%Y}; "
                            f"pasca akhir {mid:%Y}-sekarang")
    return out


# ---- Event study ---------------------------------------------------------------------------------
@dataclass
class Window:
    anchor: pd.Timestamp
    day0: pd.Timestamp          # hari bursa target pertama >= jangkar
    base: pd.Timestamp          # hari bursa target terakhir sebelum hari 0 (dasar return)
    start: pd.Timestamp
    end: pd.Timestamp
    i0: int
    truncated: bool
    days: pd.DatetimeIndex      # hari bursa target di dalam jendela


def episode_window(trading_days: pd.DatetimeIndex, anchor: str, window: tuple[int, int]) -> Window:
    """Jendela [w0, w1] hari bursa target di sekitar jangkar. Terpotong kalau data berakhir sebelum w1."""
    td = pd.DatetimeIndex(trading_days).sort_values()
    a = pd.Timestamp(anchor)
    i0 = int(td.searchsorted(a))
    if i0 >= len(td) or i0 + window[0] < 1:
        raise ValueError(f"jangkar {anchor} di luar data")
    lo, hi = i0 + window[0], i0 + window[1]
    truncated = hi >= len(td)
    hi = min(hi, len(td) - 1)
    return Window(a, td[i0], td[i0 - 1], td[lo], td[hi], i0, truncated, td[lo:hi + 1])


def episode_metrics(close: pd.Series, win: Window, trading_days: pd.DatetimeIndex) -> dict:
    """Drawdown maksimum di jendela, hari bursa (kalender target) dari hari 0 ke titik terendah sesudah jangkar,
    return ke titik terendah, dan return kumulatif sampai akhir jendela; dasar = close terakhir seri sebelum hari 0.
    Tiap seri memakai harga di hari bursanya sendiri (tanpa forward-fill)."""
    s = close.dropna()
    before = s.loc[:win.day0 - pd.Timedelta(days=1)]
    path = s.loc[win.start:win.end]
    post = s.loc[win.day0:win.end]
    if before.empty or path.empty or post.empty:
        return {"max_drawdown": np.nan, "hari_ke_titik_terendah": np.nan, "return_ke_titik_terendah": np.nan,
                "return_kumulatif_akhir": np.nan, "tanggal_titik_terendah": pd.NaT}
    base = before.iloc[-1]
    trough = post.idxmin()
    td = pd.DatetimeIndex(trading_days)
    return {"max_drawdown": float((path / path.cummax() - 1).min()),
            "hari_ke_titik_terendah": int(td.searchsorted(trough) - win.i0),
            "return_ke_titik_terendah": float(post.min() / base - 1),
            "return_kumulatif_akhir": float(post.iloc[-1] / base - 1),
            "tanggal_titik_terendah": trough}


def episode_corr(daily: pd.DataFrame, sid: str, win: Window, trading_days: pd.DatetimeIndex, pre_days: int,
                 normal_mask: pd.Series | None = None) -> dict:
    """Korelasi harian (hari yang sama, tanggal sama-sama buka) target-seri di dalam jendela vs `pre_days`
    hari bursa target sebelum jendela (tidak tumpang tindih), plus uji Fisher z; opsional vs periode normal."""
    d = daily[daily["id"] == sid]
    td = pd.DatetimeIndex(trading_days)
    lo = td.get_loc(win.start)
    pre_start = td[max(0, lo - pre_days)]
    pre_end = td[lo - 1]
    inside = d[(d["Date"] >= win.start) & (d["Date"] <= win.end)]
    pre = d[(d["Date"] >= pre_start) & (d["Date"] <= pre_end)]
    a, b = an.corr_with_ci(inside["r_x"], inside["r_target"]), an.corr_with_ci(pre["r_x"], pre["r_target"])
    z, p = an.fisher_diff(a["r"], a["n"], b["r"], b["n"])
    out = {"r_jendela": a["r"], "n_jendela": a["n"], "r_setahun_sebelum": b["r"], "n_setahun_sebelum": b["n"],
           "z_fisher": z, "p_fisher": p}
    if normal_mask is not None:
        nm = d[d["Date"].map(normal_mask).fillna(False).astype(bool)]
        out["r_normal"] = an.corr_with_ci(nm["r_x"], nm["r_target"])["r"]
    return out


def normal_day_mask(trading_days: pd.DatetimeIndex, windows: list[Window]) -> pd.Series:
    """True untuk hari bursa target di luar semua jendela episode."""
    m = pd.Series(True, index=pd.DatetimeIndex(trading_days))
    for w in windows:
        m.loc[w.start:w.end] = False
    return m


def event_study(close: pd.DataFrame, daily: pd.DataFrame, target_id: str, ids: list[str], anchors: dict[str, str],
                window: tuple[int, int], pre_days: int, normal_mask: pd.Series | None = None) -> pd.DataFrame:
    """Satu baris per (episode, seri): metrik harga untuk target & seri; korelasi harian untuk seri non-target."""
    td = close[target_id].dropna().index
    rows = []
    for name, anchor in anchors.items():
        win = episode_window(td, anchor, window)
        for sid in [target_id] + list(ids):
            row = {"episode": name, "jangkar": win.anchor, "hari_0": win.day0, "awal": win.start, "akhir": win.end,
                   "terpotong": win.truncated, "id": sid, **episode_metrics(close[sid], win, td)}
            if sid != target_id:
                row.update(episode_corr(daily, sid, win, td, pre_days, normal_mask))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_episodes(es: pd.DataFrame, exclude: list[str], cols: list[str]) -> pd.DataFrame:
    """Median lintas episode per seri, tanpa episode di `exclude`."""
    kept = es[~es["episode"].isin(exclude)]
    out = kept.groupby("id")[cols].median()
    out.attrs["episode"] = sorted(kept["episode"].unique())
    return out


def event_paths(close: pd.DataFrame, ids: list[str], win: Window, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Jalur harga (dasar = 100 di close terakhir sebelum hari 0) per hari bursa target, untuk grafik.
    Untuk gambar saja: seri yang libur di hari bursa target memakai close terakhirnya."""
    td = pd.DatetimeIndex(trading_days)
    offsets = [td.get_loc(d) - win.i0 for d in win.days]
    out = {}
    for sid in ids:
        s = close[sid].dropna()
        base = s.loc[:win.day0 - pd.Timedelta(days=1)].iloc[-1]
        out[sid] = (s.reindex(s.index.union(win.days)).ffill().reindex(win.days) / base * 100).to_numpy()
    return pd.DataFrame(out, index=pd.Index(offsets, name="hari"))


# ---- Aset pelindung ------------------------------------------------------------------------------
def crisis_week_mask(weekly_index: pd.DatetimeIndex, windows: list[Window]) -> pd.Series:
    """True untuk minggu (label Jumat) yang memuat hari bursa target di dalam salah satu jendela."""
    m = pd.Series(False, index=pd.DatetimeIndex(weekly_index))
    for w in windows:
        for d in w.days:
            m.loc[(m.index >= d) & (m.index < d + pd.Timedelta(days=7))] = True
    return m


def crisis_weekly_corr(weekly: pd.DataFrame, target_col: str, ids: list[str], crisis: pd.Series) -> pd.DataFrame:
    """Korelasi mingguan target-seri di minggu krisis vs minggu normal; uji Fisher z; apakah tandanya berbalik."""
    rows = []
    for sid in ids:
        c = crisis.reindex(weekly.index).fillna(False).astype(bool)
        a = an.corr_with_ci(weekly.loc[c, sid], weekly.loc[c, target_col])
        b = an.corr_with_ci(weekly.loc[~c, sid], weekly.loc[~c, target_col])
        z, p = an.fisher_diff(a["r"], a["n"], b["r"], b["n"])
        rows.append({"id": sid, "r_krisis": a["r"], "ci_lo_krisis": a["ci_lo"], "ci_hi_krisis": a["ci_hi"], "n_krisis": a["n"],
                     "r_normal": b["r"], "ci_lo_normal": b["ci_lo"], "ci_hi_normal": b["ci_hi"], "n_normal": b["n"],
                     "z_fisher": z, "p_fisher": p, "tanda_berbalik": bool(np.sign(a["r"]) != np.sign(b["r"]))})
    out = pd.DataFrame(rows).assign(_g=1)
    return an.add_bh(out, "p_fisher", ["_g"]).drop(columns="_g").set_index("id")


def episode_returns(close: pd.DataFrame, target_id: str, ids: list[str], transforms: dict[str, str],
                    anchors: dict[str, str], window: tuple[int, int]) -> pd.DataFrame:
    """Per episode: perubahan tiap seri dari close terakhir sebelum hari 0 sampai (a) titik terendah target
    sesudah jangkar dan (b) akhir jendela. Seri 'log' dalam % sederhana, seri 'diff' (yield) dalam poin persen."""
    td = close[target_id].dropna().index
    rows = []
    for name, anchor in anchors.items():
        win = episode_window(td, anchor, window)
        trough = close[target_id].loc[win.day0:win.end].idxmin()
        for sid in [target_id] + list(ids):
            s = close[sid].dropna()
            before = s.loc[:win.day0 - pd.Timedelta(days=1)]
            if before.empty or s.loc[:trough].empty:
                continue
            base, at_trough, at_end = before.iloc[-1], s.loc[:trough].iloc[-1], s.loc[:win.end].iloc[-1]
            if transforms.get(sid, "log") == "log":
                f = lambda v: float(v / base - 1)          # noqa: E731
            else:
                f = lambda v: float(v - base)             # noqa: E731
            rows.append({"episode": name, "id": sid, "satuan": "%" if transforms.get(sid, "log") == "log" else "pp",
                         "ke_titik_terendah_target": f(at_trough), "ke_akhir_jendela": f(at_end),
                         "tanggal_titik_terendah_target": trough, "terpotong": win.truncated})
    return pd.DataFrame(rows)


# ---- Hipotesis Brent -----------------------------------------------------------------------------
def quantile_regression(weekly: pd.DataFrame, y: str, xs: list[str], quantiles: list[float],
                        start=None, end=None) -> pd.DataFrame:
    """Regresi kuantil y ~ xs di tiap kuantil (SE robust kernel), plus OLS-HAC sebagai pembanding."""
    df = weekly.loc[start:end, [y] + list(xs)].dropna()
    X = sm.add_constant(df[list(xs)])
    rows = []
    for q in quantiles:
        fit = QuantReg(df[y], X).fit(q=q, vcov="robust", max_iter=5000)
        ci = fit.conf_int()
        for v in xs:
            rows.append({"kuantil": q, "variabel": v, "beta": fit.params[v], "ci_lo": ci.loc[v, 0],
                         "ci_hi": ci.loc[v, 1], "p": fit.pvalues[v], "n": int(len(df))})
    ols = sm.OLS(df[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": an.hac_lags(len(df))})
    ci = ols.conf_int()
    for v in xs:
        rows.append({"kuantil": "OLS", "variabel": v, "beta": ols.params[v], "ci_lo": ci.loc[v, 0],
                     "ci_hi": ci.loc[v, 1], "p": ols.pvalues[v], "n": int(len(df))})
    out = pd.DataFrame(rows)
    return an.add_bh(out, "p", ["variabel"])


def spike_interaction(weekly: pd.DataFrame, y: str, x: str, controls: list[str], threshold: float,
                      start=None, end=None) -> pd.DataFrame:
    """OLS-HAC: y ~ x + x*1[x > threshold] + 1[x > threshold] + kontrol. Koefisien interaksi menguji apakah
    hubungan y-x berbeda (atau berbalik tanda) di minggu lonjakan x. `threshold` ditetapkan dari periode lain."""
    df = weekly.loc[start:end, [y, x] + list(controls)].dropna().copy()
    df["lonjakan"] = (df[x] > threshold).astype(float)
    df[f"{x}_x_lonjakan"] = df[x] * df["lonjakan"]
    regs = [x, f"{x}_x_lonjakan", "lonjakan"] + list(controls)
    fit = sm.OLS(df[y], sm.add_constant(df[regs])).fit(cov_type="HAC", cov_kwds={"maxlags": an.hac_lags(len(df))})
    ci = fit.conf_int()
    out = pd.DataFrame({"beta": fit.params[regs], "ci_lo": ci.loc[regs, 0], "ci_hi": ci.loc[regs, 1],
                        "p": fit.pvalues[regs]})
    slope = fit.params[x] + fit.params[f"{x}_x_lonjakan"]
    w = fit.t_test(f"{x} + {x}_x_lonjakan = 0")
    out.loc[f"{x} saat lonjakan (total)"] = [slope, float(w.conf_int()[0][0]), float(w.conf_int()[0][1]), float(w.pvalue)]
    out.attrs.update({"n": int(len(df)), "n_lonjakan": int(df["lonjakan"].sum()), "ambang": threshold})
    return out


# ---- Ringkasan --------------------------------------------------------------------------------------
def phase5_summary(shift52: pd.DataFrame, med: pd.DataFrame, med_sens: pd.DataFrame, es: pd.DataFrame, qr: dict,
                   spikes: dict, havens: pd.DataFrame, haven_ret: pd.DataFrame, names: dict, target_id: str,
                   brent_id: str, brent_label: str, descriptive: list[str]) -> str:
    n = lambda i: names.get(i, i)  # noqa: E731
    L = ["### Ringkasan Phase 5 (DESKRIPTIF, tidak mengubah Spesifikasi Phase 6)", ""]
    t = med.loc[target_id]
    peers = med.drop(index=target_id)
    L.append(f"- **Seberapa dalam & cepat** (median {len(med.attrs.get('episode', []))} episode, tanpa {', '.join(descriptive)}): "
             f"{n(target_id)} drawdown maks {t.max_drawdown:.1%}, titik terendah {t.hari_ke_titik_terendah:.0f} hari bursa setelah jangkar, "
             f"return +60 hari {t.return_kumulatif_akhir:+.1%}. Pembanding: "
             + "; ".join(f"{n(i)} {r.max_drawdown:.1%} / {r.hari_ke_titik_terendah:.0f} hari / {r.return_kumulatif_akhir:+.1%}"
                         for i, r in peers.iterrows()) + ".")
    ts = med_sens.loc[target_id]
    L.append(f"  - Sensitivitas jangkar 2018 = 5 Feb: median {n(target_id)} drawdown {ts.max_drawdown:.1%}, "
             f"return +60 hari {ts.return_kumulatif_akhir:+.1%}.")
    up = peers.assign(naik=peers["r_jendela"] - peers["r_setahun_sebelum"])
    L.append("- **Korelasi saat krisis vs setahun sebelumnya** (median, harian hari yang sama): "
             + "; ".join(f"{n(i)} {r.r_setahun_sebelum:+.2f} → {r.r_jendela:+.2f}" for i, r in up.iterrows()) + ".")
    for per, tab in qr.items():
        b = tab[tab["variabel"] == brent_id].set_index("kuantil")
        L.append(f"- **Brent per kuantil ({per}, *{brent_label}*):** "
                 + ", ".join(f"τ={k}: {r.beta:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]" for k, r in b.iterrows()) + ".")
    for per, tab in spikes.items():
        r = tab.loc[[i for i in tab.index if "total" in i][0]]
        L.append(f"  - Minggu lonjakan Brent ({per}, {tab.attrs['n_lonjakan']} minggu): kemiringan total {r.beta:+.3f} "
                 f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}].")
    hv = haven_ret[~haven_ret["episode"].isin(descriptive)].groupby("id")["ke_titik_terendah_target"].median()
    L.append("- **Aset pelindung** (median perubahan dari sebelum jangkar sampai titik terendah IHSG): "
             + "; ".join(f"{n(i)} {v:+.1%}" if i in haven_ret.loc[haven_ret['satuan'] == '%', 'id'].values
                         else f"{n(i)} {v:+.2f} pp" for i, v in hv.items()) + ".")
    flips = havens[havens["tanda_berbalik"]]
    L.append("- **Korelasi mingguan dengan IHSG, minggu krisis vs normal:** "
             + "; ".join(f"{n(i)} {r.r_normal:+.2f} → {r.r_krisis:+.2f}" for i, r in havens.iterrows())
             + (f". Tanda berbalik: {', '.join(n(i) for i in flips.index)}." if len(flips) else ". Tidak ada yang berbalik tanda."))
    return "\n".join(L)
