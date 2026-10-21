"""Pemeriksaan kualitas data (Phase 1).

Isi modul:
- series_quality / quality_report : tanggal mulai, kelengkapan per tahun, gap, return nol,
                                    gerakan besar dan dugaan bad tick (ambang per kategori)
- decide_start                    : apakah target cukup lengkap sejak 2007, atau fallback ke 2010
- sanity_check_2008               : bandingkan data Yahoo dengan angka referensi IHSG 2008
- open_quality / open_module_decision : apakah data Open target layak untuk analisis gap & intraday
- phase1_summary / save_phase1_outputs : ringkasan otomatis dan file hasil untuk Phase 2
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

OPEN_METRICS = {
    "missing": "missing_pct",
    "outside_range": "outside_range_pct",
    "open_eq_prev_close": "open_eq_prev_close_pct",
    "flat_bar": "flat_bar_pct",
}
OPEN_LABELS = {
    "missing": "Open kosong",
    "outside_range": "Open di luar Low-High",
    "open_eq_prev_close": "Open = Close kemarin",
    "flat_bar": "OHLC datar",
}


# ---- Dasar --------------------------------------------------------------------
def to_returns(s: pd.Series, transform: str) -> pd.Series:
    """Log return untuk harga; selisih level untuk yield."""
    if transform == "log":
        return np.log(s.where(s > 0)).diff()
    return s.diff()


def yearly_counts(close: pd.Series, tol_days: int = 7) -> pd.Series:
    """Jumlah observasi per tahun, hanya untuk tahun yang tercakup penuh oleh data."""
    close = close.dropna()
    if close.empty:
        return pd.Series(dtype=int)
    first, last = close.index[0], close.index[-1]
    tol = pd.Timedelta(days=tol_days)
    years = [y for y in range(first.year, last.year + 1)
             if pd.Timestamp(y, 1, 1) >= first - tol and pd.Timestamp(y, 12, 31) <= last + tol]
    return close.groupby(close.index.year).size().reindex(years, fill_value=0)


def flag_moves(close: pd.Series, transform: str, threshold: float, reversal_ratio: float = 0.5) -> pd.DataFrame:
    """Gerakan harian di atas ambang. `bad_tick` = gerakan yang besoknya dibalik
    minimal `reversal_ratio` ke arah sebaliknya (pola khas harga salah satu hari)."""
    r = to_returns(close.dropna(), transform).dropna()
    nxt = r.shift(-1)
    big = r.abs() > threshold
    reverted = (np.sign(nxt) == -np.sign(r)) & (nxt.abs() >= reversal_ratio * r.abs())
    out = pd.DataFrame({"ret": r, "next_ret": nxt, "bad_tick": reverted})[big]
    out.index.name = "Date"
    return out


# ---- Kualitas per seri ----------------------------------------------------------
def series_quality(df: pd.DataFrame | None, meta: dict, requested_start: pd.Timestamp, qcfg: dict) -> dict:
    base = {k: meta[k] for k in ("id", "ticker", "name", "group", "category", "session")}
    if df is None or df.empty or "Close" not in df or df["Close"].dropna().empty:
        return {**base, "first_date": pd.NaT, "last_date": pd.NaT, "n_obs": 0, "min_obs_year": np.nan,
                "min_obs_year_at": np.nan, "max_gap_days": np.nan, "max_gap_end": pd.NaT, "pct_zero": np.nan,
                "n_large": np.nan, "n_bad_tick": np.nan, "status": "tidak tersedia"}

    close = df["Close"].dropna()
    if meta["transform"] == "log":
        close = close[close > 0]
    first, last = close.index[0], close.index[-1]
    counts = yearly_counts(close)
    gaps = close.index.to_series().diff().dt.days
    r = to_returns(close, meta["transform"]).dropna()
    moves = flag_moves(close, meta["transform"], qcfg["spike_threshold"][meta["category"]], qcfg["reversal_ratio"])
    n_bad = int(moves["bad_tick"].sum())

    issues = []
    if (first - pd.Timestamp(requested_start)).days > qcfg["start_tolerance_days"]:
        issues.append(f"mulai {first:%Y-%m-%d}")
    if len(counts) and counts.min() < qcfg["min_obs_per_year"]:
        issues.append(f"bolong ({int(counts.min())} obs di {counts.idxmin()})")
    if gaps.max() > qcfg["max_gap_days"]:
        issues.append(f"gap {int(gaps.max())} hari s/d {gaps.idxmax():%Y-%m-%d}")
    if n_bad:
        issues.append(f"{n_bad} dugaan bad tick")

    return {
        **base,
        "first_date": first,
        "last_date": last,
        "n_obs": int(len(close)),
        "min_obs_year": int(counts.min()) if len(counts) else np.nan,
        "min_obs_year_at": int(counts.idxmin()) if len(counts) else np.nan,
        "max_gap_days": int(gaps.max()) if gaps.notna().any() else np.nan,
        "max_gap_end": gaps.idxmax() if gaps.notna().any() else pd.NaT,
        "pct_zero": float((r == 0).mean() * 100) if len(r) else np.nan,
        "n_large": int(len(moves)),
        "n_bad_tick": n_bad,
        "status": "OK" if not issues else "; ".join(issues),
    }


def quality_report(prices: dict[str, pd.DataFrame], table: pd.DataFrame,
                   requested_start: pd.Timestamp, qcfg: dict) -> pd.DataFrame:
    rows = [series_quality(prices.get(m["ticker"]), m, requested_start, qcfg)
            for m in table.to_dict("records")]
    return pd.DataFrame(rows)


def moves_table(prices: dict[str, pd.DataFrame], table: pd.DataFrame, qcfg: dict) -> pd.DataFrame:
    """Semua gerakan besar lintas seri, untuk diinspeksi manual."""
    parts = []
    for m in table.to_dict("records"):
        df = prices.get(m["ticker"])
        if df is None or df.empty:
            continue
        mv = flag_moves(df["Close"], m["transform"], qcfg["spike_threshold"][m["category"]], qcfg["reversal_ratio"])
        if not mv.empty:
            parts.append(mv.reset_index().assign(id=m["id"], category=m["category"]))
    cols = ["id", "category", "Date", "ret", "next_ret", "bad_tick"]
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)[cols].sort_values(["id", "Date"]).reset_index(drop=True)


# ---- Keputusan periode -----------------------------------------------------------
def decide_start(target_close: pd.Series, preferred: str, fallback: str,
                 min_obs_per_year: int, tol_days: int = 30, label: str = "target") -> dict:
    """Pakai `preferred` (2007) hanya kalau target mulai tepat waktu DAN tahun-tahun
    sebelum `fallback` lengkap. Selain itu fallback ke 2010 (atau tanggal mulai data)."""
    close = target_close.dropna()
    if close.empty:
        raise ValueError(f"Data {label} kosong; cek koneksi lalu jalankan ulang download.")
    preferred_ts, fallback_ts = pd.Timestamp(preferred), pd.Timestamp(fallback)
    tol = pd.Timedelta(days=tol_days)
    first = close.index[0]
    counts = close.groupby(close.index.year).size()
    early_years = list(range(preferred_ts.year, fallback_ts.year))
    thin = {y: int(counts.get(y, 0)) for y in early_years if counts.get(y, 0) < min_obs_per_year}

    if first <= preferred_ts + tol and not thin:
        start_used, used_fallback = preferred_ts, False
        reason = (f"{label} tersedia sejak {first:%d %b %Y} dan {early_years[0]}-{early_years[-1]} lengkap "
                  f"-> analisis mulai {preferred_ts:%d %b %Y} (GFC 2008 tercakup)")
    else:
        why = f"data baru mulai {first:%d %b %Y}" if first > preferred_ts + tol else \
            "tahun kurang lengkap: " + ", ".join(f"{y} ({n} obs)" for y, n in thin.items())
        if first <= fallback_ts + tol:
            start_used, used_fallback = fallback_ts, True
            reason = f"{why} -> fallback ke {fallback_ts:%d %b %Y}"
        else:
            start_used, used_fallback = first.normalize(), True
            reason = f"{why}, lebih lambat dari fallback -> mulai {first:%d %b %Y}"
    return {"first_available": first, "start_used": start_used, "used_fallback": used_fallback, "reason": reason}


# ---- Sanity check 2008 -------------------------------------------------------------
def sanity_check_2008(jkse_close: pd.Series, ref: dict) -> pd.DataFrame:
    close = jkse_close.dropna()
    y08 = close.loc["2008-01-01":"2008-12-31"]
    rows = []

    def level_check(label: str, spec: dict, use_max: bool) -> None:
        d, expected, tol = pd.Timestamp(spec["date"]), float(spec["close"]), float(spec["tol_pct"])
        row = {"cek": label, "tanggal": d.date(), "referensi": f"{expected:,.2f}"}
        if y08.empty:
            rows.append({**row, "data_yahoo": "-", "selisih": "-", "status": "NO DATA", "catatan": "tidak ada data 2008"})
            return
        ext_date = y08.idxmax() if use_max else y08.idxmin()
        note = f"{'maks' if use_max else 'min'} 2008 di data: {y08[ext_date]:,.2f} pada {ext_date:%d %b %Y}"
        actual = close.get(d)
        if actual is None or pd.isna(actual):
            rows.append({**row, "data_yahoo": "-", "selisih": "-", "status": "NO DATA", "catatan": note})
            return
        diff = (actual / expected - 1) * 100
        rows.append({**row, "data_yahoo": f"{actual:,.2f}", "selisih": f"{diff:+.2f}%",
                     "status": "PASS" if abs(diff) <= tol else "FAIL", "catatan": note})

    level_check("Puncak 2008", ref["peak"], use_max=True)
    level_check("Dasar 2008", ref["trough"], use_max=False)

    rets = close.pct_change() * 100
    for spec in ref.get("daily_drops", []):
        d, expected, tol = pd.Timestamp(spec["date"]), float(spec["ret_pct"]), float(spec["tol_pp"])
        row = {"cek": "Penurunan harian", "tanggal": d.date(), "referensi": f"{expected:+.2f}%"}
        actual = rets.get(d)
        if actual is None or pd.isna(actual):
            rows.append({**row, "data_yahoo": "-", "selisih": "-", "status": "NO DATA",
                         "catatan": "tanggal tidak ada di data"})
            continue
        prev_date = close.loc[:d].index[-2]
        diff = actual - expected
        rows.append({**row, "data_yahoo": f"{actual:+.2f}%", "selisih": f"{diff:+.2f} pp",
                     "status": "PASS" if abs(diff) <= tol else "FAIL",
                     "catatan": f"dibanding close {prev_date:%d %b %Y}"})

    for day in ref.get("suspended_days", []):
        d = pd.Timestamp(day)
        row = {"cek": "Hari suspend BEI", "tanggal": d.date(), "referensi": "tidak ada perdagangan", "selisih": "-"}
        if d not in close.index:
            rows.append({**row, "data_yahoo": "tidak ada baris", "status": "PASS",
                         "catatan": "sesuai: diperlakukan sebagai hari libur"})
            continue
        prev = close.loc[:d].iloc[-2]
        if np.isclose(close[d], prev):
            rows.append({**row, "data_yahoo": f"{close[d]:,.2f} (= hari sebelumnya)", "status": "WARN",
                         "catatan": "baris basi; dibuang sebagai hari libur di Phase 2"})
        else:
            rows.append({**row, "data_yahoo": f"{close[d]:,.2f}", "status": "FAIL",
                         "catatan": "ada harga berbeda pada hari suspend; cek sumber data"})
    return pd.DataFrame(rows)


# ---- Kualitas data Open -------------------------------------------------------------
def open_quality(df: pd.DataFrame, rel_tol: float = 1e-6) -> tuple[dict, pd.DataFrame]:
    """Persentase hari bermasalah untuk empat cek data Open, keseluruhan dan per tahun."""
    d = df.dropna(subset=["Close"])
    o, h, l, c = d["Open"], d["High"], d["Low"], d["Close"]
    prev_c = c.shift(1)
    missing = o.isna() | (o <= 0)
    valid = ~missing
    outside = valid & h.notna() & l.notna() & ((o > h * (1 + rel_tol)) | (o < l * (1 - rel_tol)))
    eq_prev = valid & prev_c.notna() & ((o - prev_c).abs() <= rel_tol * prev_c.abs())
    flat = valid & ((o - c).abs() <= rel_tol * c.abs()) & ((h - l).abs() <= rel_tol * c.abs())
    flags = pd.DataFrame({"missing": missing, "outside_range": outside,
                          "open_eq_prev_close": eq_prev, "flat_bar": flat}).astype(float)
    per_year = flags.groupby(flags.index.year).mean() * 100
    per_year.insert(0, "n_days", flags.groupby(flags.index.year).size())
    per_year.index.name = "year"
    overall = {k: float(v) for k, v in (flags.mean() * 100).items()}
    overall["n_days"] = int(len(flags))
    return overall, per_year


def open_module_decision(overall: dict, per_year: pd.DataFrame, thresholds: dict, min_years: int) -> dict:
    """Pilih mode analisis return: gap+intraday penuh, sebagian (sejak tahun X), atau close-to-close."""
    thr = {m: float(thresholds[k]) for m, k in OPEN_METRICS.items()}
    failing = [m for m in thr if overall[m] > thr[m]]
    year_ok = (per_year[list(thr)] <= pd.Series(thr)).all(axis=1)

    usable_from = None  # awal periode valid terpanjang yang berakhir di tahun terakhir
    for year in sorted(year_ok.index, reverse=True):
        if not year_ok[year]:
            break
        usable_from = int(year)
    n_years = int(per_year.index.max() - usable_from + 1) if usable_from is not None else 0

    if not failing:
        return {"mode": "gap_intraday", "start_year": int(per_year.index.min()),
                "reason": "Semua cek Open di bawah batas -> return dipecah jadi gap pembukaan & intraday"}
    detail = ", ".join(f"{OPEN_LABELS[m]} {overall[m]:.1f}% (batas {thr[m]:.1f}%)" for m in failing)
    if usable_from is not None and n_years >= min_years:
        return {"mode": "gap_intraday_partial", "start_year": usable_from,
                "reason": f"Keseluruhan gagal ({detail}), tapi data Open lolos sejak {usable_from} "
                          f"({n_years} tahun) -> modul gap/intraday hanya untuk {usable_from}-sekarang"}
    return {"mode": "close_to_close", "start_year": None,
            "reason": f"Kualitas Open tidak memadai ({detail}) -> fallback ke close-to-close; "
                      f"keterbatasan dicatat di README"}


# ---- Ringkasan & output -------------------------------------------------------------
def phase1_summary(quality: pd.DataFrame, start_decision: dict, sanity: pd.DataFrame | None,
                   open_decision: dict, moves: pd.DataFrame, label: str = "^JKSE") -> str:
    """Ringkasan insight Phase 1 dalam bahasa sederhana (markdown), dibangun dari hasil."""
    lines = ["### Ringkasan Phase 1", ""]
    lines.append(f"- **Periode analisis:** {start_decision['reason']}.")

    ok = quality[quality["status"] == "OK"]
    lines.append(f"- **Kualitas data:** {len(ok)} dari {len(quality)} seri lolos semua cek.")
    for _, q in quality[quality["status"] != "OK"].iterrows():
        lines.append(f"  - {q['name']} ({q['ticker']}): {q['status']}")

    bad = moves[moves["bad_tick"].astype(bool)].groupby("id").size().sort_values(ascending=False)
    if len(bad):
        listing = ", ".join(f"{sid} ({n})" for sid, n in bad.items())
        lines.append(f"- **Dugaan bad tick** (gerakan besar yang langsung berbalik): {listing}. "
                     "Dibersihkan di Phase 2 dengan aturan yang sudah dikunci di README.")
    else:
        lines.append("- **Dugaan bad tick:** tidak ada.")

    if sanity is None or sanity.empty:
        lines.append("- **Sanity check 2008:** dilewati (tidak ada angka referensi untuk target ini).")
    else:
        n_pass = int((sanity["status"] == "PASS").sum())
        lines.append(f"- **Sanity check 2008:** {n_pass} dari {len(sanity)} cek PASS.")
        for _, s in sanity[sanity["status"] != "PASS"].iterrows():
            lines.append(f"  - {s['cek']} {s['tanggal']}: {s['status']} — {s['catatan']}")

    lines.append(f"- **Data Open {label}:** {open_decision['reason']}.")
    return "\n".join(lines)


def _json_default(obj):
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def save_phase1_outputs(out_dir: Path, quality: pd.DataFrame, moves: pd.DataFrame, sanity: pd.DataFrame | None,
                        open_per_year: pd.DataFrame, start_decision: dict, open_decision: dict,
                        target_id: str) -> list[Path]:
    """Simpan hasil Phase 1 ke reports/phase1/. decisions.json dibaca oleh Phase 2."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "quality": out_dir / "data_quality.csv",
        "moves": out_dir / "large_moves.csv",
        "sanity": out_dir / "sanity_check_2008.csv",
        "open": out_dir / "open_quality_by_year.csv",
        "decisions": out_dir / "decisions.json",
    }
    quality.to_csv(paths["quality"], index=False)
    moves.to_csv(paths["moves"], index=False)
    (sanity if sanity is not None else pd.DataFrame()).to_csv(paths["sanity"], index=False)
    open_per_year.round(3).to_csv(paths["open"])
    bad = moves[moves["bad_tick"].astype(bool)].groupby("id").size()
    decisions = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "target_id": target_id,
        "target_first_available": start_decision["first_available"],
        "analysis_start": start_decision["start_used"],
        "used_fallback": start_decision["used_fallback"],
        "start_reason": start_decision["reason"],
        "return_mode": open_decision["mode"],
        "open_module_start_year": open_decision["start_year"],
        "open_reason": open_decision["reason"],
        "bad_ticks_per_series": {k: int(v) for k, v in bad.items()},
        "unavailable_series": quality.loc[quality["status"] == "tidak tersedia", "ticker"].tolist(),
    }
    paths["decisions"].write_text(json.dumps(decisions, indent=2, default=_json_default, ensure_ascii=False),
                                  encoding="utf-8")
    return list(paths.values())
