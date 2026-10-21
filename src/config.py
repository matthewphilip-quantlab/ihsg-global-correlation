"""Membaca dan memvalidasi config/tickers.yaml."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "tickers.yaml"

GROUPS = ("target", "core", "tools", "nanti")
CATEGORIES = {"index", "etf", "fx", "sentiment", "macro", "commodity"}
SESSIONS = {"target", "sync", "after"}
TRANSFORMS = {"log", "diff"}
SOURCES = {"yfinance", "fred"}
REQUIRED = ("id", "ticker", "name", "category", "session", "transform")


def load_config(path: str | Path | None = None) -> dict:
    """Baca YAML lalu validasi strukturnya. Error-nya sengaja eksplisit supaya salah ketik cepat ketahuan."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    for key in ("project", "quality", "target", "core"):
        if key not in cfg:
            raise ValueError(f"tickers.yaml: bagian '{key}' tidak ada")
    if len(cfg["target"]) != 1:
        raise ValueError("tickers.yaml: 'target' harus berisi tepat satu seri")

    seen_ids, seen_tickers = set(), set()
    thresholds = cfg["quality"]["spike_threshold"]
    for group in GROUPS:
        for s in cfg.get(group) or []:
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

    target = cfg["target"][0]
    if target["id"] != cfg["project"]["target_id"]:
        raise ValueError("tickers.yaml: project.target_id tidak sama dengan id di 'target'")


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
                **{k: s[k] for k in REQUIRED},
            })
    cols = ["id", "ticker", "name", "group", "category", "session", "transform",
            "source", "override", "status", "note"]
    return pd.DataFrame(rows, columns=cols)


def target_ticker(cfg: dict) -> str:
    return cfg["target"][0]["ticker"]
