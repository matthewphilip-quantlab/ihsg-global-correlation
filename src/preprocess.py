"""Cleaning, alignment, dan returns (Phase 2).

Isi modul:
- holiday_rows / fx_bad_ticks / stale_quote_flags : aturan cleaning (README poin 1, 4 + Amandemen 1, 8)
- clean_prices                                    : panel close bersih + log semua penghapusan
- pair_daily_returns / weekly_returns             : return di tanggal yang sama-sama buka (tanpa forward-fill)
- jkse_in_usd                                     : IHSG dalam USD = ^JKSE / USD/IDR bersih
- usdidr_retest / retest_decision                 : uji ulang USD/IDR dengan aturan yang dikunci
- phase2_summary / save_phase2_outputs / save_processed : ringkasan dan file hasil
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .quality import _json_default, to_returns

LOG_COLS = ["id", "ticker", "Date", "close", "reason", "entry_date", "reversal_date",
            "entry_ret", "reversal_ret", "run_days"]


def load_phase1_decisions(path: str | Path) -> dict:
    """Baca reports/phase1/decisions.json. Phase 2 hanya mendukung return close-to-close."""
    dec = json.loads(Path(path).read_text(encoding="utf-8"))
    if dec["return_mode"] != "close_to_close":
        raise NotImplementedError(f"return_mode '{dec['return_mode']}' belum didukung; Phase 2 hanya close_to_close")
    return dec


# ---- Aturan cleaning ----------------------------------------------------------------
def holiday_rows(df: pd.DataFrame, suspended_days: list[str], rel_tol: float = 1e-6) -> pd.DataFrame:
    """Baris yang diperlakukan sebagai hari libur: hari suspend BEI, dan baris basi
    (Close = Close kemarin DAN Volume 0, pola Yahoo mengisi hari libur dengan harga kemarin)."""
    c = df["Close"].dropna()
    prev = c.shift(1)
    vol = df["Volume"].reindex(c.index).fillna(0) if "Volume" in df else pd.Series(0.0, index=c.index)
    stale = ((c - prev).abs() <= rel_tol * prev.abs()) & (vol == 0)
    susp = c.index.isin(pd.DatetimeIndex(pd.to_datetime(suspended_days)))
    reason = np.where(susp, "hari suspend BEI -> libur", "baris basi (Close = kemarin, Volume 0) -> libur")
    out = pd.DataFrame({"close": c, "reason": reason})[susp | stale.to_numpy()]
    out.index.name = "Date"
    return out


def fx_bad_ticks(close: pd.Series, threshold: float, reversal_ratio: float = 0.5,
                 max_days: int = 5) -> pd.DataFrame:
    """Dugaan bad tick FX (README poin 4 + Amandemen 1).

    Lonjakan |log return| > threshold di hari i yang dibalik >= reversal_ratio ke arah
    sebaliknya di hari j, dengan i < j <= i + max_days (hari bursa seri itu sendiri).
    Harga hari i s/d j-1 dianggap salah. max_days = 1 adalah aturan awal poin 4.
    Hari pembalikan tidak diperiksa lagi sebagai lonjakan baru.
    """
    s = close.dropna()
    s = s[s > 0]
    v = np.log(s).diff().to_numpy()
    idx = s.index
    rows, i = [], 1
    while i < len(v):
        if abs(v[i]) > threshold:
            for j in range(i + 1, min(i + max_days, len(v) - 1) + 1):
                if np.sign(v[j]) == -np.sign(v[i]) and abs(v[j]) >= reversal_ratio * abs(v[i]):
                    rows += [{"Date": d, "close": float(s[d]), "entry_date": idx[i], "reversal_date": idx[j],
                              "entry_ret": float(v[i]), "reversal_ret": float(v[j]), "run_days": j - i}
                             for d in idx[i:j]]
                    i = j
                    break
        i += 1
    cols = ["close", "entry_date", "reversal_date", "entry_ret", "reversal_ret", "run_days"]
    if not rows:
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="Date"))
    return pd.DataFrame(rows).set_index("Date")[cols]


def stale_quote_flags(close: pd.Series, min_count: int, tol_pct: float) -> pd.Series:
    """True untuk harga yang dekat (+-tol_pct %) dengan nilai yang persis berulang >= min_count kali.
    Hanya penanda; tidak ada yang dibuang."""
    s = close.dropna()
    flag = pd.Series(False, index=close.index)
    counts = s.value_counts()
    for value in counts[counts >= min_count].index:
        flag |= ((close / value - 1).abs() <= tol_pct / 100).fillna(False)
    return flag


# ---- Panel bersih ---------------------------------------------------------------------
def clean_prices(prices: dict[str, pd.DataFrame], table: pd.DataFrame, cfg: dict,
                 fx_max_days: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Terapkan aturan cleaning ke semua seri.

    - target: hari suspend & baris basi -> NaN (libur)
    - kategori fx: bad tick (Amandemen 1) -> NaN; kuotasi basi hanya ditandai
    - kategori lain: tidak diubah (gerakan besar hanya ditandai, README poin 8)

    Return: (close lebar kolom = id, log penghapusan, flag kuotasi basi kolom = id seri fx).
    `fx_max_days` None = Amandemen 1 dari config; isi 1 untuk aturan awal (sensitivitas).
    """
    qcfg = cfg["quality"]
    max_days = qcfg["fx_cleaning"]["max_reversal_days"] if fx_max_days is None else fx_max_days
    suspended = cfg["sanity_checks"]["jkse_2008"].get("suspended_days", [])
    closes, logs, flags = {}, [], {}
    for m in table.to_dict("records"):
        df = prices.get(m["ticker"])
        close = df["Close"].copy() if df is not None and not df.empty else pd.Series(dtype=float)
        removed = None
        if m["group"] == "target" and not close.empty:
            removed = holiday_rows(df, suspended)
        elif m["category"] == "fx" and not close.empty:
            removed = fx_bad_ticks(close, qcfg["spike_threshold"]["fx"], qcfg["reversal_ratio"], max_days)
            removed = removed.assign(reason=f"bad tick FX (pembalikan <= {max_days} hari bursa)")
        if removed is not None and len(removed):
            close.loc[removed.index] = np.nan
            logs.append(removed.reset_index().assign(id=m["id"], ticker=m["ticker"]))
        if m["category"] == "fx" and not close.empty:
            sq = qcfg["stale_quote"]
            flags[m["id"]] = stale_quote_flags(close, sq["min_count"], sq["tol_pct"])
        closes[m["id"]] = close
    log = pd.concat(logs, ignore_index=True).reindex(columns=LOG_COLS) if logs else pd.DataFrame(columns=LOG_COLS)
    panel = pd.DataFrame(closes).sort_index()
    flag_df = pd.DataFrame(flags).reindex(panel.index).fillna(False).astype(bool)
    return panel, log.sort_values(["id", "Date"]).reset_index(drop=True), flag_df


# ---- Returns ------------------------------------------------------------------------
def pair_daily_returns(close: pd.DataFrame, target_id: str, transforms: dict[str, str],
                       flags: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return harian per pasangan target-seri, hanya di tanggal yang DUA-DUANYA punya harga.
    Return setelah hari libur salah satu bursa otomatis mencakup beberapa hari (tanpa forward-fill).
    `flag_stale_quote` = True kalau salah satu ujung return seri itu adalah kuotasi basi."""
    parts = []
    for sid in close.columns:
        if sid == target_id:
            continue
        df = close[[target_id, sid]].dropna()
        if len(df) < 2:
            continue
        out = pd.DataFrame({"id": sid,
                            "r_target": to_returns(df[target_id], "log"),
                            "r_x": to_returns(df[sid], transforms[sid]),
                            "gap_days": df.index.to_series().diff().dt.days})
        f = flags[sid].reindex(df.index).fillna(False) if flags is not None and sid in flags else \
            pd.Series(False, index=df.index)
        out["flag_stale_quote"] = (f | f.shift(1, fill_value=False)).astype(bool)
        parts.append(out.iloc[1:])
    cols = ["Date", "id", "r_target", "r_x", "gap_days", "flag_stale_quote"]
    if not parts:
        return pd.DataFrame(columns=cols)
    res = pd.concat(parts).rename_axis("Date").reset_index()
    return res[cols].sort_values(["id", "Date"]).reset_index(drop=True)


def weekly_returns(close: pd.DataFrame, transforms: dict[str, str],
                   start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Return mingguan Jumat-Jumat dengan metode scripts/screening.py:
    potong periode dulu, ambil harga terakhir yang tersedia tiap minggu (W-FRI), lalu return."""
    wk = close.loc[start:end].resample("W-FRI").last()
    return pd.DataFrame({c: to_returns(wk[c], transforms[c]) for c in close.columns})


def jkse_in_usd(close: pd.DataFrame, target_id: str, fx_id: str, out_id: str) -> pd.Series:
    """IHSG dalam USD = IHSG / (IDR per USD), hanya di tanggal yang dua-duanya punya harga."""
    df = close[[target_id, fx_id]].dropna()
    return (df[target_id] / df[fx_id]).rename(out_id)


# ---- Uji ulang USD/IDR ----------------------------------------------------------------
def floor3(x: float) -> float:
    """Potong ke 3 desimal tanpa pembulatan ke atas (epsilon menahan galat float seperti 0.29999999)."""
    return math.floor(abs(x) * 1000 + 1e-9) / 1000


def weekly_corr(close: pd.DataFrame, x_id: str, target_id: str, transforms: dict[str, str],
                window: list[str], min_n: int = 52) -> tuple[float, int]:
    w = weekly_returns(close[[target_id, x_id]], transforms, window[0], window[1])
    pair = w[[x_id, target_id]].dropna()
    r = float(pair[x_id].corr(pair[target_id])) if len(pair) >= min_n else np.nan
    return r, int(len(pair))


def usdidr_retest(variants: dict[str, tuple[pd.DataFrame, bool]], x_id: str, target_id: str,
                  transforms: dict[str, str], window: list[str], min_abs: float) -> pd.DataFrame:
    """Satu baris per varian data: {nama: (panel close, mengikat?)}."""
    rows = []
    for name, (close, binding) in variants.items():
        r, n = weekly_corr(close, x_id, target_id, transforms, window)
        rows.append({"varian": name, "r": r, "abs_r_3dp": floor3(r), "n_minggu": n,
                     "lolos": floor3(r) >= min_abs, "mengikat": binding})
    return pd.DataFrame(rows)


def retest_decision(retest: pd.DataFrame, min_abs: float, old_name: str) -> dict:
    """Keputusan dari baris yang mengikat, plus pembanding aturan lama."""
    b = retest[retest["mengikat"]].iloc[0]
    old = retest[retest["varian"] == old_name].iloc[0]
    status = "driver" if b["lolos"] else "tool_permanen"
    verdict = "LOLOS -> driver (Core)" if b["lolos"] else "TIDAK LOLOS -> permanen alat bantu"
    reason = (f"|r| mingguan 2010-2019 pasca-cleaning ({b['varian']}) = {b['abs_r_3dp']:.3f} "
              f"(r = {b['r']:+.4f}, {b['n_minggu']} minggu), batas {min_abs:.3f}: {verdict}")
    differs = bool(old["lolos"] != b["lolos"])
    note = (f"PERHATIAN: status berbeda dengan aturan lama ({old['abs_r_3dp']:.3f} -> "
            f"{'lolos' if old['lolos'] else 'tidak lolos'})." if differs else
            f"Status sama dengan aturan lama ({old['abs_r_3dp']:.3f}; sensitivitas, tidak mengikat).")
    return {"status": status, "passed": bool(b["lolos"]), "abs_r": float(b["abs_r_3dp"]), "r": float(b["r"]),
            "n_weeks": int(b["n_minggu"]), "differs_from_old_rule": differs, "reason": reason, "note": note}


# ---- Ringkasan & output ---------------------------------------------------------------
def pair_coverage(daily: pd.DataFrame) -> pd.DataFrame:
    """Jumlah return harian per pasangan, rentang tanggal, dan berapa yang mencakup > 1 hari bursa."""
    g = daily.groupby("id")
    return pd.DataFrame({"n_return": g.size(), "mulai": g["Date"].min().dt.date, "akhir": g["Date"].max().dt.date,
                         "pct_gap_gt4": (g["gap_days"].apply(lambda s: (s > 4).mean() * 100)).round(2),
                         "n_flag_stale": g["flag_stale_quote"].sum().astype(int)})


def phase2_summary(log: pd.DataFrame, flags: pd.DataFrame, decision: dict, coverage: pd.DataFrame,
                   moves_other: pd.DataFrame, fx_id: str, weekly_flagged: int) -> str:
    lines = ["### Ringkasan Phase 2", ""]
    by_reason = log.groupby(["id", "reason"]).size()
    lines.append("- **Data yang dibuang** (detail di `reports/phase2/cleaning_log.csv`):")
    for (sid, reason), n in by_reason.items():
        lines.append(f"  - {sid}: {n} hari, {reason}")
    n_flag = int(flags[fx_id].sum()) if fx_id in flags else 0
    lines.append(f"- **Kuotasi basi {fx_id}** yang tersisa dan hanya ditandai: {n_flag} hari "
                 f"({weekly_flagged} di antaranya menjadi harga mingguan). Diputuskan sebelum Phase 4.")
    lines.append(f"- **Uji ulang {fx_id}:** {decision['reason']}. {decision['note']}")
    lines.append(f"- **Gerakan besar non-FX** ({len(moves_other)}) hanya ditandai, tidak ada yang dibuang.")
    lines.append(f"- **Return harian:** {len(coverage)} pasangan IHSG-seri, "
                 f"{int(coverage['n_return'].min()):,}-{int(coverage['n_return'].max()):,} return per pasangan, "
                 "hanya di tanggal yang sama-sama buka.")
    return "\n".join(lines)


def save_phase2_outputs(out_dir: Path, log: pd.DataFrame, retest: pd.DataFrame, flagged: pd.DataFrame,
                        coverage: pd.DataFrame, moves_other: pd.DataFrame, decision: dict,
                        extra: dict) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "log": out_dir / "cleaning_log.csv",
        "retest": out_dir / "usdidr_retest.csv",
        "flags": out_dir / "stale_quote_flags.csv",
        "coverage": out_dir / "pair_coverage.csv",
        "moves": out_dir / "large_moves_non_fx.csv",
        "decisions": out_dir / "decisions.json",
    }
    log.to_csv(paths["log"], index=False)
    retest.round({"r": 4}).to_csv(paths["retest"], index=False)
    flagged.to_csv(paths["flags"])
    coverage.to_csv(paths["coverage"])
    moves_other.to_csv(paths["moves"], index=False)
    decisions = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                 "usdidr_retest": decision, **extra}
    paths["decisions"].write_text(json.dumps(decisions, indent=2, default=_json_default, ensure_ascii=False),
                                  encoding="utf-8")
    return list(paths.values())


def save_processed(out_dir: Path, frames: dict[str, pd.DataFrame | pd.Series]) -> list[Path]:
    """Simpan panel ke data/processed/ (tidak di-commit)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame in frames.items():
        p = out_dir / f"{name}.parquet"
        (frame.to_frame() if isinstance(frame, pd.Series) else frame).to_parquet(p)
        paths.append(p)
    return paths
