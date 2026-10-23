"""Forward test: model beku + log prediksi yang dicatat SEBELUM hasilnya terjadi.

Prediksi di log ini adalah satu-satunya bahan evaluasi jujur untuk versi model mana pun di masa depan:
parameternya dibekukan pada tanggal tertentu, dan setiap baris dicatat sebelum hari target terjadi.

Isi modul:
- freeze_models / freeze_comparators / save_frozen / load_frozen : model beku (logit utama & GARCH-X) dan pembanding
                                             beku (baseline naif & GARCH tanpa VIX) dari data latih yang sama
- next_signal / signal_window              : probabilitas IHSG naik & prakiraan volatilitas untuk hari bursa berikutnya,
                                             dan jendela waktu kapan prediksi sah dicatat
- schedule_check                           : apakah jam jadwal lokal selalu jatuh di dalam jendela (DST AS & lokal)
- append_log / fill_outcomes / on_time / evaluate_log : log append-only, isi hasil aktual, evaluasi pra-registrasi
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import oos
from . import volatility as vol

LOG_COLS = ["model_version", "logged_at_utc", "feature_date", "p_naik", "sigma_pct", "p_naik_baseline",
            "sigma_pct_tanpa_vix", "fitur_json", "data_terakhir_target", "target_date", "r_aktual_pct", "y_aktual"]
MIN_ROWS_EVAL = 250          # pra-registrasi: evaluasi pertama setelah 250 baris valid, tidak lebih awal


# ---- Membekukan model -------------------------------------------------------------------------------
def build_inputs(cfg: dict, close: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.Series, pd.Series]:
    """Frame fitur (sama dengan Phase 6, model utama), daftar fitur, return target (%) dan dlog VIX harian."""
    ocfg = cfg["oos"]
    tid, fx_id, vix_id = cfg["project"]["target_id"], cfg["usdidr_retest"]["id"], ocfg["vix_id"]
    transforms = {s["id"]: s["transform"] for g in ("core", "tools") for s in cfg.get(g) or []}
    frame = oos.feature_frame(close, tid, ocfg["main_features"], transforms, fx_id=fx_id, eido_id=ocfg["eido_id"])
    feats = [f"r_{i}" for i in ocfg["main_features"]] + ["d_premium", "r_target"]
    ret = (np.log(close[tid].dropna()).diff() * 100).dropna()
    x = oos.own_returns(close[vix_id]).reindex(ret.index).fillna(0.0)
    return frame, feats, ret, x


def _training_data(cfg: dict, close: pd.DataFrame, as_of: str):
    as_of_ts = pd.Timestamp(as_of)
    frame, feats, ret, x = build_inputs(cfg, close.loc[:as_of_ts])
    train = frame.loc[cfg["oos"]["train_start"]:]
    train = train[train["target_date"] <= as_of_ts].dropna(subset=feats + ["y"])
    r, xx = ret.loc[cfg["oos"]["train_start"]:as_of_ts], x.loc[cfg["oos"]["train_start"]:as_of_ts]
    return as_of_ts, feats, train, r, xx


def freeze_models(cfg: dict, close: pd.DataFrame, as_of: str) -> dict:
    """Logit utama (MLE, tanpa regularisasi) pada baris yang targetnya terealisasi <= as_of, dan GARCH-X pada return
    train_start..as_of. Tidak ada tuning; spesifikasi = Phase 6."""
    as_of_ts, feats, train, r, xx = _training_data(cfg, close, as_of)
    logit = oos._fit_logit(train[feats], train["y"])
    g = vol.fit_garchx(r.to_numpy(), xx.to_numpy())
    return {"model_version": f"frozen_{as_of_ts:%Y-%m-%d}", "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "as_of": f"{as_of_ts:%Y-%m-%d}", "train_start": cfg["oos"]["train_start"], "n_train_logit": int(len(train)),
            "logit_features": feats, "logit_params": {k: float(v) for k, v in logit.params.items()},
            "garch_params": dict(zip(g["names"], map(float, g["params"]))), "garch_converged": g["converged"],
            "n_train_garch": int(len(r)), "spesifikasi": "README 'Spesifikasi Phase 6 (dikunci sebelum Phase 5)', model utama"}


def freeze_comparators(cfg: dict, close: pd.DataFrame, as_of: str, model: dict) -> dict:
    """Pembanding beku untuk evaluasi forward (pra-registrasi): baseline naif = frekuensi naik di data latih logit
    beku, dan GARCH(1,1) tanpa VIX pada return yang sama. Data latih harus identik dengan model beku."""
    as_of_ts, feats, train, r, xx = _training_data(cfg, close, as_of)
    if len(train) != model["n_train_logit"] or len(r) != model["n_train_garch"]:
        raise ValueError("data latih tidak identik dengan model beku; pembanding tidak dibuat")
    g0 = vol.fit_garchx(r.to_numpy(), xx.to_numpy(), gamma_fixed=0.0)
    return {"model_version": model["model_version"], "file": f"{model['model_version']}_pembanding",
            "as_of": f"{as_of_ts:%Y-%m-%d}", "baseline_p_naik": float(train["y"].mean()), "n_train_logit": int(len(train)),
            "garch_tanpa_vix_params": dict(zip(g0["names"], map(float, g0["params"]))), "n_train_garch": int(len(r)),
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def save_frozen(model: dict, directory: Path) -> Path:
    path = Path(directory) / f"{model.get('file', model['model_version'])}.json"
    if path.exists():
        raise FileExistsError(f"{path} sudah ada; model beku tidak boleh ditimpa")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_frozen(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---- Sinyal hari bursa berikutnya --------------------------------------------------------------------
def _garch_next(params: dict, r: np.ndarray, x: np.ndarray) -> float:
    pvec = np.array([params["mu"], params["omega"], params["alpha"], params["beta"], params["gamma"]])
    s2, e = vol.garchx_sigma2(pvec, r, x)
    return float(np.sqrt((params["omega"] + params["alpha"] * e[-1] ** 2 + params["beta"] * s2[-1])
                         * np.exp(params["gamma"] * x[-1])))


def next_signal(cfg: dict, close: pd.DataFrame, model: dict, comparators: dict | None = None) -> dict:
    """Sinyal untuk hari bursa IHSG sesudah tanggal terakhir IHSG di data.
    Ditolak (ValueError) kalau fitur sesi AS untuk tanggal itu belum lengkap."""
    frame, feats, ret, x = build_inputs(cfg, close)
    tid = cfg["project"]["target_id"]
    last_target = close[tid].dropna().index.max()
    row = frame.loc[last_target, feats] if last_target in frame.index else None
    if row is None or row.isna().any():
        missing = [] if row is None else [f for f in feats if pd.isna(row[f])]
        raise ValueError(f"fitur untuk {last_target:%Y-%m-%d} belum lengkap {missing}: tunggu data sesi AS tanggal itu")
    params = pd.Series(model["logit_params"])
    z = params["const"] + float((params[feats] * row[feats]).sum())
    p_up = float(1 / (1 + np.exp(-z)))
    r = ret.loc[cfg["oos"]["train_start"]:last_target].to_numpy()
    xx = x.loc[cfg["oos"]["train_start"]:last_target].to_numpy()
    out = {"model_version": model["model_version"], "feature_date": last_target, "p_naik": p_up,
           "sigma_pct": _garch_next(model["garch_params"], r, xx), "fitur": {f: float(row[f]) for f in feats},
           "p_naik_baseline": np.nan, "sigma_pct_tanpa_vix": np.nan}
    if comparators is not None:
        out["p_naik_baseline"] = comparators["baseline_p_naik"]
        out["sigma_pct_tanpa_vix"] = _garch_next(comparators["garch_tanpa_vix_params"], r, xx)
    return out


def schedule_check(hour: int, minute: int, tz: str, start: str, end: str, weekdays=(0, 1, 2, 3, 4),
                   us_close_utc: str = "21:30", open_hour_utc: int = 2) -> pd.DataFrame:
    """Untuk tiap hari jadwal (zona waktu lokal `tz`) di [start, end]: tanggal fitur yang relevan (hari kerja
    terakhir yang sesi AS-nya sudah final) dan apakah jam itu di dalam signal_window. Memperhitungkan DST lokal
    dan AS karena batas jendela ditulis dalam UTC yang konservatif."""
    from zoneinfo import ZoneInfo
    hh, mm = map(int, us_close_utc.split(":"))
    rows = []
    for d in pd.date_range(start, end, freq="D"):
        if d.dayofweek not in weekdays:
            continue
        run = pd.Timestamp(d.year, d.month, d.day, hour, minute, tz=ZoneInfo(tz)).tz_convert("UTC")
        f = (run - pd.Timedelta(hours=hh, minutes=mm)).tz_localize(None).normalize()
        while f.dayofweek >= 5:
            f -= pd.Timedelta(days=1)
        ok, why = signal_window(f, run, us_close_utc, open_hour_utc)
        start_w = pd.Timestamp(f"{f:%Y-%m-%d} {us_close_utc}", tz="UTC")
        end_w = (f + pd.offsets.BDay(1)).tz_localize("UTC") + pd.Timedelta(hours=open_hour_utc)
        rows.append({"jadwal_lokal": d.date(), "run_utc": run, "feature_date": f.date(), "ok": ok,
                     "menit_setelah_batas_awal": (run - start_w).total_seconds() / 60,
                     "menit_sebelum_bei_buka": (end_w - run).total_seconds() / 60})
    return pd.DataFrame(rows)


def signal_window(feature_date: pd.Timestamp, now_utc: pd.Timestamp, us_close_utc: str = "21:30",
                  open_hour_utc: int = 2) -> tuple[bool, str]:
    """Prediksi untuk fitur tanggal t hanya sah di antara (a) setelah sesi AS tanggal t final, yaitu t 21:30 UTC
    (16:00 ET musim dingin = 21:00 UTC, plus jeda), dan (b) sebelum BEI buka di hari kerja berikutnya (09:00 WIB = 02:00 UTC).
    Sebelum (a), Yahoo bisa mengirim bar AS yang masih berjalan; sesudah (b), hasilnya mulai terlihat."""
    now = pd.Timestamp(now_utc)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    start = pd.Timestamp(f"{pd.Timestamp(feature_date):%Y-%m-%d} {us_close_utc}", tz="UTC")
    end = (pd.Timestamp(feature_date) + pd.offsets.BDay(1)).tz_localize("UTC") + pd.Timedelta(hours=open_hour_utc)
    if now < start:
        return False, f"sesi AS {pd.Timestamp(feature_date):%Y-%m-%d} belum final (tunggu s/d {start:%Y-%m-%d %H:%M} UTC)"
    if now >= end:
        return False, f"terlambat: BEI sudah buka {end:%Y-%m-%d %H:%M} UTC; hari target tidak lagi di masa depan"
    return True, "dalam jendela"


# ---- Log append-only ------------------------------------------------------------------------------------
def read_log(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLS)
    return pd.read_csv(path, parse_dates=["feature_date", "target_date"])


def append_log(signal: dict, path: Path, last_target_date: pd.Timestamp, now_utc: str | None = None) -> bool:
    """Tambah satu baris kalau (model_version, feature_date) belum ada. Return True kalau baris ditambahkan.
    Baris lama tidak pernah diubah oleh fungsi ini."""
    path = Path(path)
    log = read_log(path)
    key = (log["model_version"] == signal["model_version"]) & (pd.to_datetime(log["feature_date"]) == signal["feature_date"])
    if key.any():
        return False
    row = {"model_version": signal["model_version"],
           "logged_at_utc": now_utc or datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "feature_date": signal["feature_date"].strftime("%Y-%m-%d"), "p_naik": round(signal["p_naik"], 6),
           "sigma_pct": round(signal["sigma_pct"], 6), "p_naik_baseline": round(float(signal["p_naik_baseline"]), 6),
           "sigma_pct_tanpa_vix": round(float(signal["sigma_pct_tanpa_vix"]), 6), "fitur_json": json.dumps(signal["fitur"]),
           "data_terakhir_target": last_target_date.strftime("%Y-%m-%d"), "target_date": "", "r_aktual_pct": "", "y_aktual": ""}
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=LOG_COLS).to_csv(path, mode="a", header=not path.exists(), index=False)
    return True


def fill_outcomes(path: Path, target_close: pd.Series) -> int:
    """Isi target_date / r_aktual_pct / y_aktual untuk baris yang hari targetnya sudah terjadi. Kolom prediksi
    tidak disentuh. Return jumlah baris yang baru diisi."""
    path = Path(path)
    log = read_log(path)
    if log.empty:
        return 0
    s = target_close.dropna()
    filled = 0
    for i, r in log.iterrows():
        if pd.notna(r["y_aktual"]) and str(r["y_aktual"]) != "":
            continue
        later = s.index[s.index > pd.Timestamp(r["feature_date"])]
        if len(later) == 0 or pd.Timestamp(r["feature_date"]) not in s.index:
            continue
        td = later[0]
        ret = float(np.log(s[td] / s[pd.Timestamp(r["feature_date"])]) * 100)
        log.loc[i, ["target_date", "r_aktual_pct", "y_aktual"]] = [td, round(ret, 6), float(ret > 0)]
        filled += 1
    if filled:
        log.to_csv(path, index=False)
    return filled


def on_time(log: pd.DataFrame, open_hour_utc: int = 2) -> pd.Series:
    """True kalau baris dicatat sebelum bursa buka (09:00 WIB = 02:00 UTC) pada hari kerja pertama sesudah
    feature_date. Batas ini konservatif: kalau hari itu libur, prediksi tetap harus tercatat sebelum tanggal itu."""
    deadline = (pd.to_datetime(log["feature_date"]) + pd.offsets.BDay(1)).dt.tz_localize("UTC") + pd.Timedelta(hours=open_hour_utc)
    logged = pd.to_datetime(log["logged_at_utc"], utc=True)
    return logged < deadline


def evaluate_log(path: Path, model_version: str | None = None, min_rows: int = MIN_ROWS_EVAL) -> pd.DataFrame:
    """Evaluasi PRA-REGISTRASI (README "Forward test"): hanya baris yang tepat waktu dan hasilnya sudah terjadi.
    Sebelum `min_rows` baris valid, hanya jumlah baris yang dilaporkan, tanpa metrik dan uji.
    Arah: log loss, DM (log loss) vs baseline naif beku; volatilitas: QLIKE & MSE, DM vs GARCH tanpa VIX beku."""
    log = read_log(path)
    log = log[pd.to_numeric(log["y_aktual"], errors="coerce").notna()]
    log = log[on_time(log)] if len(log) else log
    if model_version:
        log = log[log["model_version"] == model_version]
    rows = []
    for mv, d in log.groupby("model_version"):
        if len(d) < min_rows:
            rows.append({"model_version": mv, "n_valid": int(len(d)),
                         "status": f"belum dievaluasi: {len(d)} < {min_rows} baris valid (pra-registrasi)"})
            continue
        y, p = d["y_aktual"].astype(float).to_numpy(), d["p_naik"].astype(float).to_numpy()
        pb = d["p_naik_baseline"].astype(float).to_numpy()
        r2 = d["r_aktual_pct"].astype(float).to_numpy() ** 2
        s2, s2b = d["sigma_pct"].astype(float).to_numpy() ** 2, d["sigma_pct_tanpa_vix"].astype(float).to_numpy() ** 2
        dm_dir = oos.diebold_mariano(oos.log_loss_obs(y, p), oos.log_loss_obs(y, pb))
        dm_q = oos.diebold_mariano(vol.qlike(r2, s2), vol.qlike(r2, s2b))
        dm_m = oos.diebold_mariano(vol.mse(r2, s2), vol.mse(r2, s2b))
        m, mb = oos.metrics(y, p), oos.metrics(y, pb)
        rows.append({"model_version": mv, "n_valid": int(len(d)), "status": "dievaluasi", **m,
                     "log_loss_baseline": mb["log_loss"], "hit_rate_baseline": mb["hit_rate"],
                     "dm_arah_p": dm_dir["p"], "dm_arah_selisih": dm_dir["selisih_loss_rata"],
                     "qlike_vix": float(vol.qlike(r2, s2).mean()), "qlike_tanpa_vix": float(vol.qlike(r2, s2b).mean()),
                     "dm_qlike_p": dm_q["p"], "dm_mse_p": dm_m["p"]})
    return pd.DataFrame(rows)
