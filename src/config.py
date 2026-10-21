"""Membaca dan memvalidasi config/tickers.yaml.

Target analisis dipilih dari katalog `targets` lewat `project.target_id` (atau argumen
`target_id` di load_config). Setelah dibaca, cfg["target"] berisi tepat satu seri: target aktif.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "tickers.yaml"

GROUPS = ("target", "core", "tools", "nanti")
CATEGORIES = {"index", "etf", "fx", "sentiment", "macro", "commodity", "stock"}
SESSIONS = {"target", "sync", "after"}
TRANSFORMS = {"log", "diff"}
SOURCES = {"yfinance", "fred"}
REQUIRED = ("id", "ticker", "name", "category", "session", "transform")


def load_config(path: str | Path | None = None, target_id: str | None = None) -> dict:
    """Baca YAML, pilih target aktif, lalu validasi strukturnya.
    Error-nya sengaja eksplisit supaya salah ketik cepat ketahuan."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = resolve_target(cfg, target_id)
    validate_config(cfg)
    return cfg


def resolve_target(cfg: dict, target_id: str | None = None) -> dict:
    """Isi cfg["target"] dengan seri dari katalog `targets` yang id-nya = target_id
    (default project.target_id). Mengembalikan salinan; cfg asli tidak diubah."""
    cfg = copy.deepcopy(cfg)
    if target_id is not None:
        cfg["project"]["target_id"] = target_id
    tid = cfg["project"]["target_id"]
    matches = [s for s in cfg.get("targets") or [] if s["id"] == tid]
    if len(matches) != 1:
        raise ValueError(f"tickers.yaml: target_id '{tid}' harus muncul tepat sekali di blok 'targets'")
    cfg["target"] = [matches[0]]
    return cfg


def validate_config(cfg: dict) -> None:
    for key in ("project", "quality", "targets", "target", "core"):
        if key not in cfg:
            raise ValueError(f"tickers.yaml: bagian '{key}' tidak ada")
    if len(cfg["target"]) != 1:
        raise ValueError("tickers.yaml: 'target' harus berisi tepat satu seri")

    seen_ids, seen_tickers = set(), set()
    thresholds = cfg["quality"]["spike_threshold"]
    catalog = [("targets", s) for s in cfg["targets"]]
    others = [(g, s) for g in GROUPS if g != "target" for s in cfg.get(g) or []]
    for group, s in catalog + others:
        where = f"tickers.yaml [{group}] {s.get('id', '?')}"
        missing = [k for k in REQUIRED if k not in s]
        if missing:
            raise ValueError(f"{where}: field wajib hilang {missing}")
        if s["id"] in seen_ids:
            raise ValueError(f"{where}: id duplikat")
        if s["ticker"] in seen_tickers:
            raise ValueError(f"{where}: ticker {s['ticker']} muncul lebih dari sekali")
        seen_ids.add(s["id"])
        seen_tickers.add(s["ticker"])
        if s["category"] not in CATEGORIES:
            raise ValueError(f"{where}: category '{s['category']}' tidak dikenal")
        if s["category"] not in thresholds:
            raise ValueError(f"{where}: belum ada spike_threshold untuk '{s['category']}'")
        if s["session"] not in SESSIONS:
            raise ValueError(f"{where}: session '{s['session']}' tidak dikenal")
        if s["transform"] not in TRANSFORMS:
            raise ValueError(f"{where}: transform '{s['transform']}' tidak dikenal")
        if s.get("source", "yfinance") not in SOURCES:
            raise ValueError(f"{where}: source '{s['source']}' tidak dikenal")
        if group == "targets" and s["session"] != "target":
            raise ValueError(f"{where}: seri di katalog 'targets' harus session 'target'")

    target = cfg["target"][0]
    if target["id"] != cfg["project"]["target_id"] or target not in cfg["targets"]:
        raise ValueError("tickers.yaml: 'target' tidak sama dengan project.target_id di katalog 'targets'")


def series_table(cfg: dict, groups: tuple[str, ...] = ("target", "core", "tools")) -> pd.DataFrame:
    """Semua seri dari grup yang diminta sebagai satu tabel (satu baris per seri)."""
    rows = []
    for group in groups:
        for s in cfg.get(group) or []:
            rows.append({
                "group": group,
                "source": s.get("source", "yfinance"),
                "override": bool(s.get("override", False)),
                "status": s.get("status", ""),
                "note": s.get("note", s.get("reason", "")),
                "label": s.get("label", ""),
                **{k: s[k] for k in REQUIRED},
            })
    cols = ["id", "ticker", "name", "group", "category", "session", "transform",
            "source", "override", "status", "note", "label"]
    return pd.DataFrame(rows, columns=cols)


def target_ticker(cfg: dict) -> str:
    return cfg["target"][0]["ticker"]


def sanity_ref(cfg: dict) -> dict | None:
    """Angka referensi 2008 untuk target aktif plus hari suspend BEI; None kalau target tidak punya referensi."""
    ref = (cfg.get("sanity_checks") or {}).get(cfg["project"]["target_id"])
    if ref is None:
        return None
    return {**ref, "suspended_days": cfg["project"].get("suspended_days", [])}


def target_usd_id(cfg: dict) -> str:
    """Nama kolom target dalam USD, mis. JKSE_USD."""
    return cfg["project"]["target_id"] + cfg["target_usd"]["suffix"]


def output_dir(cfg: dict, base: str, *parts: str, root: Path = PROJECT_ROOT) -> Path:
    """Folder output: <root>/<base>/<parts> untuk primary_target, dan
    <root>/<base>/targets/<target_id>/<parts> untuk target lain (hasil IHSG tidak tertimpa)."""
    tid = cfg["project"]["target_id"]
    if tid == cfg["project"].get("primary_target", tid):
        return Path(root, base, *parts)
    return Path(root, base, "targets", tid, *parts)
