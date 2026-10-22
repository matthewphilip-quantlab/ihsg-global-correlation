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
    for ax in list(axes.flat)[len(eps):]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower left")
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
