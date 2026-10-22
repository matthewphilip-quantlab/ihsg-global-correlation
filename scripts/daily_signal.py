"""Forward test harian: sinyal IHSG untuk hari bursa berikutnya dari model BEKU.

Pemakaian (lihat README "Forward test"):
    python scripts/daily_signal.py --freeze            # sekali: bekukan model dari data/processed (Phase 6)
    python scripts/daily_signal.py                     # tiap hari bursa, antara ±04:30 WIB (sesi AS final) dan 09:00 WIB
    python scripts/daily_signal.py --dry-run           # hitung dan tampilkan saja, tanpa menulis log
    python scripts/daily_signal.py --evaluate          # metrik forward dari baris yang tepat waktu & sudah terjadi

Log reports/forward_test/log.csv hanya ditambah, tidak pernah ditulis ulang, kecuali kolom hasil aktual yang diisi
setelah hari target terjadi. Jangan mengedit baris prediksi secara manual.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import forward as fw  # noqa: E402
from src import leadlag as ll  # noqa: E402
from src import preprocess as pp  # noqa: E402
from src.config import load_config, output_dir, series_table  # noqa: E402
from src.data_loader import load_prices  # noqa: E402

MODELS = ROOT / "models"
LOG = ROOT / "reports" / "forward_test" / "log.csv"
WIB = timezone(timedelta(hours=7))
MAX_STALE_DAYS = 4          # data IHSG terakhir harus sebaru ini (hari kalender), kalau tidak sinyal ditolak


def latest_model() -> Path:
    files = sorted(MODELS.glob("frozen_*.json"))
    if not files:
        raise SystemExit("Belum ada model beku. Jalankan dulu: python scripts/daily_signal.py --freeze")
    return files[-1]


def fresh_close(cfg: dict) -> pd.DataFrame:
    """Download ulang (refresh) lalu terapkan aturan cleaning yang sama dengan Phase 2 & 4."""
    table = series_table(cfg)
    prices = load_prices(table["ticker"].tolist(), start=cfg["project"]["start_date"], refresh=True)
    clean, _, flags = pp.clean_prices(prices, table, cfg)
    return ll.drop_flagged(clean, flags)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze", nargs="?", const="auto", help="bekukan model per tanggal (default: akhir data/processed)")
    ap.add_argument("--model", type=Path, help="file model beku (default: yang terbaru di models/)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config()

    if args.freeze:
        panel = pd.read_parquet(output_dir(cfg, "data/processed") / "prices_clean.parquet")
        flags = pd.read_parquet(output_dir(cfg, "data/processed") / "stale_quote_flags.parquet")
        close = ll.drop_flagged(panel, flags)
        as_of = close.index.max().strftime("%Y-%m-%d") if args.freeze == "auto" else args.freeze
        path = fw.save_frozen(fw.freeze_models(cfg, close, as_of), MODELS)
        print(f"Model beku disimpan: {path.relative_to(ROOT)}")
        return 0

    if args.evaluate:
        print(fw.evaluate_log(LOG).to_string(index=False) if LOG.exists() else "Log belum ada.")
        return 0

    model = fw.load_frozen(args.model or latest_model())
    close = fresh_close(cfg)
    tid = cfg["project"]["target_id"]
    last = close[tid].dropna().index.max()
    today_wib = datetime.now(WIB).date()
    if (pd.Timestamp(today_wib) - last).days > MAX_STALE_DAYS:
        raise SystemExit(f"Data IHSG terakhir {last:%Y-%m-%d} terlalu lama (> {MAX_STALE_DAYS} hari): sinyal ditolak.")
    try:
        sig = fw.next_signal(cfg, close, model)
    except ValueError as exc:
        raise SystemExit(f"Sinyal belum bisa dibuat: {exc}")
    ok, why = fw.signal_window(sig["feature_date"], pd.Timestamp.now(tz="UTC"))
    tag = "" if ok else "  [SEMENTARA: " + why + "]"
    print(f"Model {sig['model_version']} | fitur per {sig['feature_date']:%Y-%m-%d} (IHSG tutup + sesi AS){tag}")
    print(f"  P(IHSG naik di hari bursa berikutnya) = {sig['p_naik']:.3f}")
    print(f"  Prakiraan volatilitas harian (1 sd)     = {sig['sigma_pct']:.2f}%")
    if args.dry_run:
        print("--dry-run: log tidak ditulis.")
        return 0
    if not ok:
        raise SystemExit(f"Tidak dicatat: {why}.")
    added = fw.append_log(sig, LOG, last)
    filled = fw.fill_outcomes(LOG, close[tid])
    print(f"Log: {'baris baru ditambahkan' if added else 'baris untuk tanggal ini sudah ada (tidak ditimpa)'}; "
          f"hasil aktual baru diisi: {filled}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
