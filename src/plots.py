"""Grafik untuk notebook. Setiap fungsi mengembalikan Figure (tidak memanggil plt.show)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .quality import OPEN_LABELS, OPEN_METRICS

# Palet kategorikal tetap (urutan dipakai apa adanya, tidak diputar)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def _style(ax, title: str, ylabel: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_jkse_2008(jkse_close: pd.Series, ref: dict, start: str = "2007-01-01", end: str = "2009-12-31"):
    """IHSG 2007-2009 dengan titik referensi puncak/dasar dan masa suspend Oktober 2008."""
    s = jkse_close.dropna().loc[start:end]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(s.index, s.values, color=SERIES[0], linewidth=2, label="IHSG (Yahoo)")
    pts = [(pd.Timestamp(ref[k]["date"]), ref[k]["close"]) for k in ("peak", "trough")]
    ax.scatter([d for d, _ in pts], [v for _, v in pts], s=64, color=SERIES[1], edgecolor="white",
               linewidth=2, zorder=4, label="Angka referensi (puncak & dasar)")
    for (d, v), label in zip(pts, ("Puncak", "Dasar")):
        ax.annotate(f"{label}: {v:,.0f}\n{d:%d %b %Y}", (d, v), textcoords="offset points", xytext=(14, 0),
                    va="center", fontsize=9, color=INK_2, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GRID))
    ax.margins(y=0.12)
    days = [pd.Timestamp(x) for x in ref.get("suspended_days", [])]
    if days:
        ax.axvspan(min(days), max(days) + pd.Timedelta(days=1), color=GRID, alpha=0.9, label="Suspend BEI")
    _style(ax, "IHSG 2007-2009: validasi data Yahoo terhadap angka referensi", "Indeks")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    return fig


def plot_open_quality(per_year: pd.DataFrame, thresholds: dict, label: str = "^JKSE"):
    """Persentase hari bermasalah per tahun untuk tiap cek Open, dengan garis batasnya."""
    fig, axes = plt.subplots(1, len(OPEN_METRICS), figsize=(12, 3.4), sharex=True)
    for ax, (metric, key), color in zip(axes, OPEN_METRICS.items(), SERIES):
        ax.bar(per_year.index, per_year[metric], color=color, width=0.7)
        ax.axhline(thresholds[key], color=INK_2, linestyle="--", linewidth=1)
        ax.text(per_year.index.min(), thresholds[key], f" batas {thresholds[key]:.0f}%",
                va="bottom", fontsize=8, color=INK_2)
        _style(ax, OPEN_LABELS[metric], "% hari" if metric == "missing" else "")
        ax.tick_params(axis="x", rotation=90)
    fig.suptitle(f"Kualitas data Open {label} per tahun", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    return fig


def plot_flagged_moves(close: pd.Series, moves: pd.DataFrame, title: str):
    """Harga dengan tanda pada dugaan bad tick, untuk inspeksi visual."""
    s = close.dropna()
    bad = moves[moves["bad_tick"].astype(bool)]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(s.index, s.values, color=SERIES[0], linewidth=1.5, label="Close")
    if len(bad):
        ax.scatter(bad.index, s.reindex(bad.index), s=64, color=SERIES[1], edgecolor="white",
                   linewidth=2, zorder=3, label=f"Dugaan bad tick ({len(bad)})")
    _style(ax, title)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    return fig


def plot_fx_cleaning(raw: pd.Series, clean: pd.Series, title: str, start: str | None = None,
                     end: str | None = None):
    """Harga FX sebelum dan sesudah cleaning; titik = harga yang dibuang."""
    raw, clean = raw.dropna().loc[start:end], clean.loc[start:end]
    removed = raw[clean.reindex(raw.index).isna()]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(raw.index, raw.values, color=GRID, linewidth=1.5, label="Sebelum cleaning")
    ax.plot(clean.index, clean.values, color=SERIES[0], linewidth=1.5, label="Sesudah cleaning")
    if len(removed):
        ax.scatter(removed.index, removed.values, s=40, color=SERIES[1], edgecolor="white",
                   linewidth=1.5, zorder=3, label=f"Dibuang ({len(removed)})")
    _style(ax, title)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    return fig


def plot_idr_vs_usd(target: pd.Series, target_usd: pd.Series, title: str, name: str = "IHSG"):
    """Target dalam rupiah vs dalam USD, keduanya diindeks = 100 di tanggal bersama pertama."""
    df = pd.concat([target.rename("idr"), target_usd.rename("usd")], axis=1).dropna()
    df = df / df.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(df.index, df["idr"], color=SERIES[0], linewidth=1.5, label=f"{name} (rupiah)")
    ax.plot(df.index, df["usd"], color=SERIES[1], linewidth=1.5, label=f"{name} dalam USD")
    _style(ax, title, f"Indeks ({df.index[0]:%b %Y} = 100)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    return fig


# Diverging biru <-> merah dengan titik tengah abu-abu netral (korelasi -1..+1)
DIVERGING = ["#1c5cab", "#86b6ef", "#f0efec", "#f19a99", "#b83a39"]


def plot_corr_heatmap(matrix: pd.DataFrame, title: str):
    """Matriks korelasi dengan skala diverging tetap -1..+1 dan nilai tertulis di tiap sel."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    n = len(matrix)
    fig, ax = plt.subplots(figsize=(0.62 * n + 2.4, 0.55 * n + 1.4))
    im = ax.imshow(matrix.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(n), matrix.columns, rotation=45, ha="right", fontsize=9, color=INK_2)
    ax.set_yticks(range(n), matrix.index, fontsize=9, color=INK_2)
    for i in range(n):
        for j in range(n):
            v = matrix.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.6 else INK)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=8, colors=INK_2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    return fig


def plot_corr_by_period(tab: pd.DataFrame, names: dict, title: str, periods=("screening", "pasca", "penuh")):
    """Korelasi per seri dengan CI 95%, satu warna per periode (urutan warna tetap)."""
    t = tab.set_index(["id", "periode"])
    order = (tab[tab["periode"] == periods[0]].assign(a=lambda d: d["r"].abs()).sort_values("a")["id"].tolist())
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(order) + 1.4))
    offsets = np.linspace(-0.22, 0.22, len(periods))
    for color, per, off in zip(SERIES, periods, offsets):
        for k, sid in enumerate(order):
            if (sid, per) not in t.index:
                continue
            row = t.loc[(sid, per)]
            ax.plot([row["ci_lo"], row["ci_hi"]], [k + off] * 2, color=color, linewidth=2, solid_capstyle="round")
            ax.scatter(row["r"], k + off, s=40, color=color, edgecolor="white", linewidth=1.5, zorder=3,
                       label=per if k == 0 else None)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(range(len(order)), [names.get(i, i) for i in order])
    _style(ax, title, "")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Korelasi Pearson dengan IHSG (CI 95%)", color=INK_2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    return fig


def plot_asymmetric(asym: pd.DataFrame, names: dict, title: str):
    """Korelasi di minggu pengkondisi turun vs naik, satu baris per seri."""
    a = asym.sort_values("r_turun")
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(a) + 1.4))
    for k, r in enumerate(a.itertuples()):
        ax.plot([r.r_naik, r.r_turun], [k, k], color=GRID, linewidth=2, zorder=1)
    ax.scatter(a["r_turun"], range(len(a)), s=48, color=SERIES[1], edgecolor="white", linewidth=1.5, zorder=3,
               label="Minggu turun")
    ax.scatter(a["r_naik"], range(len(a)), s=48, color=SERIES[0], edgecolor="white", linewidth=1.5, zorder=3,
               label="Minggu naik")
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(range(len(a)), [names.get(i, i) for i in a["id"]])
    _style(ax, title, "")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Korelasi Pearson dengan IHSG", color=INK_2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    return fig


def plot_ccf_heatmap(ccf: pd.DataFrame, names: dict, sessions: dict, title: str):
    """Korelasi IHSG(t+k) vs seri(t) per lag: baris = seri (dikelompokkan per sesi), kolom = k."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    p = ccf.pivot(index="id", columns="k", values="r")
    order = sorted(p.index, key=lambda i: (sessions.get(i, ""), -p.loc[i].abs().max()))
    p = p.loc[order]
    lim = max(0.1, float(np.nanmax(np.abs(p.to_numpy()))))
    fig, ax = plt.subplots(figsize=(0.7 * p.shape[1] + 3.2, 0.48 * p.shape[0] + 1.6))
    im = ax.imshow(p.to_numpy(), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(p.shape[1]), [f"{k:+d}" for k in p.columns], fontsize=9, color=INK_2)
    ax.set_yticks(range(p.shape[0]), [f"{names.get(i, i)} ({sessions.get(i, '')})" for i in p.index],
                  fontsize=9, color=INK_2)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            v = p.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 * lim else INK)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("k (hari bursa bersama): IHSG(t+k) vs seri(t); k > 0 = seri mendahului IHSG", color=INK_2)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=8, colors=INK_2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    return fig


def plot_spread_z(z: pd.Series, title: str):
    """z-score spread EIDO vs IHSG dalam USD, dengan pita +-2."""
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.axhspan(-2, 2, color=GRID, alpha=0.5, linewidth=0, label="±2 simpangan baku")
    ax.plot(z.index, z.values, color=SERIES[0], linewidth=1.2, label="z-score spread")
    ax.axhline(0, color=INK_2, linewidth=1)
    _style(ax, title, "z-score")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    fig.tight_layout()
    return fig


def plot_rolling_corr(rolls: dict[int, pd.DataFrame], names: dict, shade: list[tuple], title: str, ncols: int = 2):
    """Small multiples: korelasi bergulir per seri (satu warna per panjang jendela), episode krisis diarsir."""
    ids = list(next(iter(rolls.values())).columns)
    nrows = int(np.ceil(len(ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.1 * nrows + 0.8), sharex=True, sharey=True)
    for ax, sid in zip(axes.flat, ids):
        for a, b in shade:
            ax.axvspan(a, b, color=GRID, alpha=0.9, linewidth=0)
        for color, (w, roll) in zip(SERIES, rolls.items()):
            ax.plot(roll.index, roll[sid], color=color, linewidth=1.5, label=f"{w} minggu")
        ax.axhline(0, color=INK_2, linewidth=0.8)
        _style(ax, names.get(sid, sid))
        ax.title.set_fontsize(10)
    for ax in list(axes.flat)[len(ids):]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    return fig


def plot_event_paths(paths: dict[str, pd.DataFrame], names: dict, title: str, ncols: int = 4, notes: dict | None = None):
    """Satu panel per episode: harga (sebelum hari 0 = 100) target dan pembanding, hari 0 = jangkar."""
    eps = list(paths)
    nrows = int(np.ceil(len(eps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.8 * nrows + 0.8), sharey=True, squeeze=False)
    for ax, ep in zip(axes.flat, eps):
        df = paths[ep]
        ax.axvline(0, color=INK_2, linewidth=0.8)
        ax.axhline(100, color=GRID, linewidth=0.8)
        for color, sid in zip(SERIES, df.columns):
            ax.plot(df.index, df[sid], color=color, linewidth=1.8 if sid == df.columns[0] else 1.2,
                    label=names.get(sid, sid))
        _style(ax, f"{ep}" + (f" ({notes[ep]})" if notes and ep in notes else ""))
        ax.title.set_fontsize(10)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    spare = list(axes.flat)[len(eps):]
    for ax in spare:
        ax.set_visible(False)
    if spare:                                   # legenda di slot panel kosong supaya tidak menimpa data
        spare[0].set_visible(True)
        spare[0].axis("off")
        spare[0].legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK_2, loc="center")
    else:
        fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.supxlabel("Hari bursa sejak jangkar", fontsize=9, color=INK_2)
    fig.tight_layout()
    return fig


def plot_quantile_coefs(tabs: dict[str, pd.DataFrame], var: str, title: str):
    """Koefisien satu variabel per kuantil (CI 95%), satu warna per periode; garis putus = OLS."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for k, (color, (per, tab)) in enumerate(zip(SERIES, tabs.items())):
        t = tab[tab["variabel"] == var]
        q = t[t["kuantil"] != "OLS"]
        x = q["kuantil"].astype(float) + (k - 0.5) * 0.012
        ax.vlines(x, q["ci_lo"], q["ci_hi"], color=color, linewidth=2)
        ax.plot(x, q["beta"], "o-", color=color, markersize=6, markeredgecolor="white", linewidth=1.2, label=per)
        ols = t[t["kuantil"] == "OLS"]["beta"].iloc[0]
        ax.axhline(ols, color=color, linestyle="--", linewidth=1)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xlabel("Kuantil return mingguan IHSG (garis putus = OLS)", color=INK_2)
    _style(ax, title, "Koefisien")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    return fig


def plot_calibration(cals: dict[str, pd.DataFrame], title: str):
    """Probabilitas prediksi rata-rata vs frekuensi naik aktual per kelompok; garis diagonal = kalibrasi sempurna."""
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    lo = min(min(c["p_rata"].min(), c["frekuensi_naik"].min()) for c in cals.values()) - 0.03
    hi = max(max(c["p_rata"].max(), c["frekuensi_naik"].max()) for c in cals.values()) + 0.03
    ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1, linestyle="--", label="Kalibrasi sempurna")
    for color, (name, c) in zip(SERIES, cals.items()):
        ax.plot(c["p_rata"], c["frekuensi_naik"], "o-", color=color, markersize=7, markeredgecolor="white",
                linewidth=1.5, label=name)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Probabilitas naik yang diprediksi (rata-rata kelompok)", color=INK_2)
    _style(ax, title, "Frekuensi naik aktual")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    return fig


def plot_bigcap_betas(table: pd.DataFrame, pairs: dict[str, tuple[str, str]], names: dict, title: str,
                      period_labels: tuple[str, str] = ("2010-2019", "2020+")):
    """Satu panel per faktor: beta tiap saham di dua periode (warna tetap per periode)."""
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 0.45 * len(table) + 1.8), sharey=True)
    ids = list(table.index)
    for ax, (label, cols) in zip(np.atleast_1d(axes), pairs.items()):
        for color, col, off, lab in zip(SERIES, cols, (-0.12, 0.12), period_labels):
            ax.scatter(table[col], np.arange(len(ids)) + off, s=48, color=color, edgecolor="white", linewidth=1.5,
                       zorder=3, label=lab)
        ax.axvline(0, color=INK_2, linewidth=0.8)
        ax.set_yticks(range(len(ids)), [names.get(i, i) for i in ids])
        _style(ax, label)
        ax.title.set_fontsize(10)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    return fig


# ---- Grafik tambahan untuk Lampiran Teknis (hanya menampilkan hasil yang sudah ada) ----------------------
SEQUENTIAL = ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


def plot_availability(counts: pd.DataFrame, title: str, expected: int = 245):
    """Jumlah hari berharga per seri per tahun (baris = seri, kolom = tahun), skala satu warna."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(counts) + 1.6))
    im = ax.imshow(counts.to_numpy(dtype=float), cmap=cmap, vmin=0, vmax=max(expected + 20, counts.max().max()), aspect="auto")
    ax.set_xticks(range(counts.shape[1]), [str(c) for c in counts.columns], rotation=90, fontsize=8, color=INK_2)
    ax.set_yticks(range(counts.shape[0]), counts.index, fontsize=9, color=INK_2)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            v = counts.iat[i, j]
            ax.text(j, i, "" if pd.isna(v) else f"{int(v)}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > 0.6 * expected else INK)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.ax.tick_params(labelsize=8, colors=INK_2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    return fig


def plot_return_distribution(r: pd.Series, title: str):
    """Histogram return harian (dengan kurva normal bermean & sd sama) dan QQ plot terhadap normal."""
    from scipy import stats
    r = r.dropna()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.hist(r, bins=120, density=True, color=SERIES[0], alpha=0.85, edgecolor="white", linewidth=0.3, label="Return harian")
    x = np.linspace(r.min(), r.max(), 400)
    a1.plot(x, stats.norm.pdf(x, r.mean(), r.std()), color=SERIES[1], linewidth=2, label="Normal (mean & sd sama)")
    a1.set_yscale("log")
    a1.set_ylim(1e-3, None)
    _style(a1, "Distribusi (skala log)", "Densitas")
    a1.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    (osm, osr), (slope, icpt, _) = stats.probplot(r, dist="norm")
    a2.scatter(osm, osr, s=6, color=SERIES[0], alpha=0.6, linewidths=0)
    a2.plot(osm, slope * osm + icpt, color=SERIES[1], linewidth=1.5, label="Garis normal")
    a2.set_xlabel("Kuantil teoretis normal", color=INK_2)
    _style(a2, "QQ plot vs normal", "Kuantil sampel")
    a2.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    return fig


def plot_beta_forest(coef: pd.DataFrame, names: dict, title: str, single: pd.DataFrame | None = None):
    """Beta multi-faktor dengan CI 95% (dan beta satu faktor sebagai pembanding, kalau ada)."""
    c = coef.sort_values("beta")
    y = np.arange(len(c))
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(c) + 1.5))
    ax.hlines(y, c["ci_lo"], c["ci_hi"], color=SERIES[0], linewidth=2)
    ax.scatter(c["beta"], y, s=48, color=SERIES[0], edgecolor="white", linewidth=1.5, zorder=3, label="Multi-faktor (CI 95% HAC)")
    if single is not None:
        s = single.reindex(c.index)
        ax.scatter(s["beta_1f"], y + 0.22, s=36, marker="D", color=SERIES[1], edgecolor="white", linewidth=1.2,
                   zorder=3, label="Satu faktor")
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_yticks(y, [names.get(i, i) for i in c.index])
    _style(ax, title)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Beta (perubahan % IHSG per 1% perubahan seri, mingguan)", color=INK_2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    return fig


def plot_premium(premium: pd.Series, title: str):
    """Level premium log EIDO - log IHSG dalam USD."""
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(premium.index, premium.values, color=SERIES[0], linewidth=1.2)
    _style(ax, title, "log EIDO − log IHSG(USD)")
    fig.tight_layout()
    return fig


def plot_haven_bars(values: pd.Series, names: dict, title: str, highlight: str | None = None):
    """Batang horizontal perubahan (%) per aset; batang `highlight` diberi warna kedua."""
    v = values.sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(v) + 1.3))
    colors = [SERIES[1] if i == highlight else SERIES[0] for i in v.index]
    ax.barh(range(len(v)), v.values * 100, color=colors, height=0.6)
    for k, x in enumerate(v.values * 100):
        ax.text(x + (0.4 if x >= 0 else -0.4), k, f"{x:+.1f}%", va="center", ha="left" if x >= 0 else "right",
                fontsize=8, color=INK)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_yticks(range(len(v)), [names.get(i, i) for i in v.index])
    lim = max(abs(v.values * 100)) * 1.35
    ax.set_xlim(-lim, lim)
    _style(ax, title)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Median perubahan (%)", color=INK_2)
    fig.tight_layout()
    return fig


def plot_walkforward_scheme(refits: pd.DatetimeIndex, train_start: str, title: str, every: int = 12):
    """Skema walk-forward: tiap baris = satu titik refit; batang latih (expanding) dan bulan uji."""
    import matplotlib.dates as md
    r = pd.DatetimeIndex(sorted(set(refits)))
    show = list(r[::every]) + ([r[-1]] if r[-1] not in r[::every] else [])
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(show) + 1.4))
    t0 = pd.Timestamp(train_start)
    for k, m0 in enumerate(show):
        m1 = m0 + pd.offsets.MonthBegin(1)
        ax.barh(k, (m0 - t0).days, left=md.date2num(t0), height=0.55, color=SERIES[0], alpha=0.35)
        ax.barh(k, (m1 - m0).days, left=md.date2num(m0), height=0.55, color=SERIES[1])
    ax.set_yticks(range(len(show)), [f"refit {m:%b %Y}" for m in show], fontsize=8)
    ax.invert_yaxis()
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(md.DateFormatter("%Y"))
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SERIES[0], alpha=0.35, label="Data latih (expanding sejak 2010)"),
                       Patch(color=SERIES[1], label="Bulan uji (prediksi OOS)")],
              frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left", bbox_to_anchor=(0, -0.07), ncol=2)
    _style(ax, title)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_vol_forecast(df: pd.DataFrame, title: str, start: str | None = None):
    """|return| harian realisasi vs prakiraan sd GARCH-X dan GARCH tanpa VIX (OOS)."""
    d = df.loc[start:]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(d.index, np.sqrt(d["r2"]), width=1.0, color=GRID, label="|return| realisasi")
    ax.plot(d.index, np.sqrt(d["s2_tanpa_vix"]), color=SERIES[0], linewidth=1.1, label="GARCH tanpa VIX (prakiraan sd)")
    ax.plot(d.index, np.sqrt(d["s2_vix"]), color=SERIES[1], linewidth=1.1, label="GARCH-X dengan VIX (prakiraan sd)")
    _style(ax, title, "% per hari")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper right")
    fig.tight_layout()
    return fig


def plot_placebo(placebo: pd.DataFrame, real: dict, title: str):
    """Hasil uji placebo target acak (titik) vs model asli (garis), untuk selisih log loss dan hit rate."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
    for ax, (col, lab, fmt) in zip(axes, [("selisih_log_loss", "Selisih log loss vs naif (negatif = lebih baik)", "{:+.4f}"),
                                          ("hit_rate", "Hit rate", "{:.1%}")]):
        rng = np.random.default_rng(0)
        ax.scatter(placebo[col], rng.uniform(-0.2, 0.2, len(placebo)), s=40, color=SERIES[0], edgecolor="white",
                   linewidth=1.2, label="Placebo (target diacak)")
        ax.axvline(real[col], color=SERIES[1], linewidth=2, label="Model asli")
        ax.text(real[col], 0.3, " " + fmt.format(real[col]), color=INK, fontsize=8, va="bottom")
        ax.set_ylim(-0.5, 0.6)
        ax.set_yticks([])
        _style(ax, lab)
        ax.title.set_fontsize(10)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK_2, loc="lower center", ncol=2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def plot_bigcap_drawdown(values: pd.Series, names: dict, title: str, reference: tuple[str, float] | None = None):
    """Median drawdown krisis per saham (batang), dengan garis referensi (mis. IHSG)."""
    v = values.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(v) + 1.4))
    ax.barh(range(len(v)), v.values * 100, color=SERIES[0], height=0.6)
    for k, x in enumerate(v.values * 100):
        ax.text(-0.6, k, f"{x:.1f}%", va="center", ha="right", fontsize=8, color="white")
    if reference is not None:
        ax.axvline(reference[1] * 100, color=SERIES[1], linewidth=2, linestyle="--", label=f"{reference[0]} {reference[1]:.1%}")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower left")
    ax.set_yticks(range(len(v)), [names.get(i, i) for i in v.index])
    ax.set_xlim(min(v.values * 100) * 1.25, 0)
    _style(ax, title)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Median drawdown maksimum di jendela krisis (%)", color=INK_2)
    fig.tight_layout()
    return fig
