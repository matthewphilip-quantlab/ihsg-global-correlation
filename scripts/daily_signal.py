"""Forward test harian: sinyal IHSG untuk hari bursa berikutnya dari model BEKU.

Pemakaian (lihat README "Forward test"):
    python scripts/daily_signal.py --freeze            # sekali: bekukan model + pembanding dari data/processed
    python scripts/daily_signal.py                     # tiap hari bursa, antara ±04:30 WIB (sesi AS final) dan 09:00 WIB
    python scripts/daily_signal.py --dry-run           # hitung dan tampilkan saja, tanpa menulis log
    python scripts/daily_signal.py --evaluate          # evaluasi pra-registrasi (baru berisi uji setelah 250 baris valid)
    python scripts/daily_signal.py --check-schedule    # cek jam jadwal lokal terhadap jendela waktu, 3 tahun ke depan

Kode keluar: 0 = sinyal dicatat ATAU dilewati dengan alasan wajar (libur, belum final, terlambat, sudah ada);
1 = error yang perlu dicek (data semua basi, download gagal, model tidak ada).

Log reports/forward_test/log.csv hanya ditambah, tidak pernah ditulis ulang, kecuali kolom hasil aktual yang diisi
setelah hari target terjadi. Jangan mengedit baris prediksi secara manual.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
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
MAX_STALE_DAYS = 4          # seri TERBARU di data harus sebaru ini (hari kalender); kalau tidak = error data
SCHEDULE = (9, 15, "Australia/Melbourne")   # jam jadwal launchd (waktu lokal Mac)


def say(msg: str) -> None:
    print(f"[{datetime.now(WIB):%Y-%m-%d %H:%M} WIB] {msg}", flush=True)


def latest_model() -> Path:
    files = sorted(f for f in MODELS.glob("frozen_*.json") if not f.stem.endswith("_pembanding"))
    if not files:
        raise SystemExit("Belum ada model beku. Jalankan dulu: python scripts/daily_signal.py --freeze")
    return files[-1]


def comparators_for(model_path: Path) -> dict:
    path = model_path.with_name(model_path.stem + "_pembanding.json")
    if not path.exists():
        raise SystemExit(f"Pembanding beku {path.name} tidak ada. Jalankan: python scripts/daily_signal.py --freeze")
    return fw.load_frozen(path)


def freeze(cfg: dict, as_of_arg: str) -> None:
    panel = pd.read_parquet(output_dir(cfg, "data/processed") / "prices_clean.parquet")
    flags = pd.read_parquet(output_dir(cfg, "data/processed") / "stale_quote_flags.parquet")
    close = ll.drop_flagged(panel, flags)
    as_of = close.index.max().strftime("%Y-%m-%d") if as_of_arg == "auto" else as_of_arg
    model = fw.freeze_models(cfg, close, as_of)
    path = MODELS / f"{model['model_version']}.json"
    if path.exists():                       # jangan menimpa; cek bahwa pembekuan ulang mereproduksi model tersimpan
        saved = fw.load_frozen(path)
        same = all(np.isclose(saved["logit_params"][k], v, rtol=1e-6) for k, v in model["logit_params"].items()) and \
            all(np.isclose(saved["garch_params"][k], v, rtol=1e-4) for k, v in model["garch_params"].items())
        if not same:
            raise SystemExit(f"Pembekuan ulang TIDAK mereproduksi {path.name}; data berubah? Tidak ada yang ditulis.")
        say(f"{path.name} sudah ada dan terreproduksi; tidak ditimpa.")
        model = saved
    else:
        say(f"Model beku disimpan: {fw.save_frozen(model, MODELS).relative_to(ROOT)}")
    comp_path = MODELS / f"{model['model_version']}_pembanding.json"
    if comp_path.exists():
        say(f"{comp_path.name} sudah ada; tidak ditimpa.")
    else:
        say(f"Pembanding beku disimpan: {fw.save_frozen(fw.freeze_comparators(cfg, close, as_of, model), MODELS).relative_to(ROOT)}")


def fresh_close(cfg: dict) -> pd.DataFrame:
    """Download ulang (refresh) lalu terapkan aturan cleaning yang sama dengan Phase 2 & 4."""
    table = series_table(cfg)
    prices = load_prices(table["ticker"].tolist(), start=cfg["project"]["start_date"], refresh=True, verbose=False)
    clean, _, flags = pp.clean_prices(prices, table, cfg)
    return ll.drop_flagged(clean, flags)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze", nargs="?", const="auto", help="bekukan model per tanggal (default: akhir data/processed)")
    ap.add_argument("--model", type=Path, help="file model beku (default: yang terbaru di models/)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--check-schedule", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config()

    if args.freeze:
        freeze(cfg, args.freeze)
        return 0
    if args.evaluate:
        print(fw.evaluate_log(LOG).to_string(index=False) if LOG.exists() and len(fw.read_log(LOG)) else "Log belum berisi baris.")
        return 0
    if args.check_schedule:
        today = pd.Timestamp.now().normalize()
        chk = fw.schedule_check(*SCHEDULE, str(today.date()), str((today + pd.DateOffset(years=3)).date()))
        print(f"Jadwal {SCHEDULE[0]:02d}:{SCHEDULE[1]:02d} {SCHEDULE[2]}, Senin-Jumat, {len(chk)} run: "
              f"semua di dalam jendela = {bool(chk['ok'].all())}; margin minimum setelah batas awal "
              f"{chk['menit_setelah_batas_awal'].min():.0f} menit, sebelum BEI buka {chk['menit_sebelum_bei_buka'].min():.0f} menit")
        return 0 if chk["ok"].all() else 1

    model_path = args.model or latest_model()
    model, comp = fw.load_frozen(model_path), comparators_for(model_path)
    try:
        close = fresh_close(cfg)
    except Exception as exc:                                  # koneksi / Yahoo
        say(f"ERROR download: {exc!r}")
        return 1
    tid = cfg["project"]["target_id"]
    newest = close.apply(lambda s: s.last_valid_index()).max()
    if (pd.Timestamp(datetime.now(WIB).date()) - newest).days > MAX_STALE_DAYS:
        say(f"ERROR: data terbaru {newest:%Y-%m-%d} lebih tua dari {MAX_STALE_DAYS} hari (download bermasalah?).")
        return 1
    try:
        sig = fw.next_signal(cfg, close, model, comp)
    except ValueError as exc:                                 # libur AS di hari bursa IHSG terakhir, atau data AS belum ada
        say(f"LEWATI: {exc}.")
        return 0
    ok, why = fw.signal_window(sig["feature_date"], pd.Timestamp.now(tz="UTC"))
    tag = "" if ok else f"  [SEMENTARA: {why}]"
    say(f"Model {sig['model_version']} | fitur per {sig['feature_date']:%Y-%m-%d} (IHSG tutup + sesi AS){tag}")
    print(f"  P(IHSG naik di hari bursa berikutnya) = {sig['p_naik']:.3f}   (baseline naif {sig['p_naik_baseline']:.3f})")
    print(f"  Prakiraan volatilitas harian (1 sd)     = {sig['sigma_pct']:.2f}%  (GARCH tanpa VIX {sig['sigma_pct_tanpa_vix']:.2f}%)")
    if args.dry_run:
        say("--dry-run: log tidak ditulis.")
        return 0
    if not ok:                                                # libur BEI, akhir pekan panjang, Mac bangun terlambat, dll.
        say(f"LEWATI (tidak dicatat): {why}.")
        return 0
    added = fw.append_log(sig, LOG, close[tid].dropna().index.max())
    filled = fw.fill_outcomes(LOG, close[tid])
    say(f"{'DICATAT' if added else 'LEWATI: baris untuk tanggal fitur ini sudah ada (tidak ditimpa)'}; "
        f"hasil aktual baru diisi: {filled}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
