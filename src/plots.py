"""Grafik untuk notebook. Setiap fungsi mengembalikan Figure (tidak memanggil plt.show)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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


def plot_open_quality(per_year: pd.DataFrame, thresholds: dict):
    """Persentase hari bermasalah per tahun untuk tiap cek Open, dengan garis batasnya."""
    fig, axes = plt.subplots(1, len(OPEN_METRICS), figsize=(12, 3.4), sharex=True)
    for ax, (metric, key), color in zip(axes, OPEN_METRICS.items(), SERIES):
        ax.bar(per_year.index, per_year[metric], color=color, width=0.7)
        ax.axhline(thresholds[key], color=INK_2, linestyle="--", linewidth=1)
        ax.text(per_year.index.min(), thresholds[key], f" batas {thresholds[key]:.0f}%",
                va="bottom", fontsize=8, color=INK_2)
        _style(ax, OPEN_LABELS[metric], "% hari" if metric == "missing" else "")
        ax.tick_params(axis="x", rotation=90)
    fig.suptitle("Kualitas data Open ^JKSE per tahun", x=0.01, ha="left", fontsize=12, color=INK)
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


def plot_idr_vs_usd(jkse: pd.Series, jkse_usd: pd.Series, title: str):
    """IHSG dalam rupiah vs dalam USD, keduanya diindeks = 100 di tanggal bersama pertama."""
    df = pd.concat([jkse.rename("idr"), jkse_usd.rename("usd")], axis=1).dropna()
    df = df / df.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(df.index, df["idr"], color=SERIES[0], linewidth=1.5, label="IHSG (rupiah)")
    ax.plot(df.index, df["usd"], color=SERIES[1], linewidth=1.5, label="IHSG dalam USD")
    _style(ax, title, f"Indeks ({df.index[0]:%b %Y} = 100)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    return fig
