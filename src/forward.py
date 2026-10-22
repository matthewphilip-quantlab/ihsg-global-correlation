"""Forward test: model beku + log prediksi yang dicatat SEBELUM hasilnya terjadi.

Prediksi di log ini adalah satu-satunya bahan evaluasi jujur untuk versi model mana pun di masa depan:
parameternya dibekukan pada tanggal tertentu, dan setiap baris dicatat sebelum hari target terjadi.

Isi modul:
- freeze_models / save_frozen / load_frozen : latih logit utama & GARCH-X sekali, simpan parameter ke JSON
- next_signal / signal_window              : probabilitas IHSG naik & prakiraan volatilitas untuk hari bursa berikutnya,
                                             dan jendela waktu kapan prediksi sah dicatat
- append_log / fill_outcomes / on_time / evaluate_log : log append-only, isi hasil aktual, metrik forward
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

LOG_COLS = ["model_version", "logged_at_utc", "feature_date", "p_naik", "sigma_pct", "fitur_json",
            "data_terakhir_target", "target_date", "r_aktual_pct", "y_aktual"]


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


def freeze_models(cfg: dict, close: pd.DataFrame, as_of: str) -> dict:
    """Logit utama (MLE, tanpa regularisasi) pada baris yang targetnya terealisasi <= as_of, dan GARCH-X pada return
    train_start..as_of. Tidak ada tuning; spesifikasi = Phase 6."""
    as_of_ts = pd.Timestamp(as_of)
    frame, feats, ret, x = build_inputs(cfg, close.loc[:as_of_ts])
    train = frame.loc[cfg["oos"]["train_start"]:]
    train = train[train["target_date"] <= as_of_ts].dropna(subset=feats + ["y"])
    logit = oos._fit_logit(train[feats], train["y"])
    r, xx = ret.loc[cfg["oos"]["train_start"]:as_of_ts], x.loc[cfg["oos"]["train_start"]:as_of_ts]
    g = vol.fit_garchx(r.to_numpy(), xx.to_numpy())
    return {"model_version": f"frozen_{as_of_ts:%Y-%m-%d}", "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "as_of": f"{as_of_ts:%Y-%m-%d}", "train_start": cfg["oos"]["train_start"], "n_train_logit": int(len(train)),
            "logit_features": feats, "logit_params": {k: float(v) for k, v in logit.params.items()},
            "garch_params": dict(zip(g["names"], map(float, g["params"]))), "garch_converged": g["converged"],
            "n_train_garch": int(len(r)), "spesifikasi": "README 'Spesifikasi Phase 6 (dikunci sebelum Phase 5)', model utama"}


def save_frozen(model: dict, directory: Path) -> Path:
    path = Path(directory) / f"{model['model_version']}.json"
    if path.exists():
        raise FileExistsError(f"{path} sudah ada; model beku tidak boleh ditimpa")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_frozen(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---- Sinyal hari bursa berikutnya --------------------------------------------------------------------
def next_signal(cfg: dict, close: pd.DataFrame, model: dict) -> dict:
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
    gp = model["garch_params"]
    pvec = np.array([gp["mu"], gp["omega"], gp["alpha"], gp["beta"], gp["gamma"]])
    r = ret.loc[cfg["oos"]["train_start"]:last_target].to_numpy()
    xx = x.loc[cfg["oos"]["train_start"]:last_target].to_numpy()
    s2, e = vol.garchx_sigma2(pvec, r, xx)
    s2_next = (gp["omega"] + gp["alpha"] * e[-1] ** 2 + gp["beta"] * s2[-1]) * np.exp(gp["gamma"] * xx[-1])
    return {"model_version": model["model_version"], "feature_date": last_target, "p_naik": p_up,
            "sigma_pct": float(np.sqrt(s2_next)), "fitur": {f: float(row[f]) for f in feats}}


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
           "sigma_pct": round(signal["sigma_pct"], 6), "fitur_json": json.dumps(signal["fitur"]),
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


def evaluate_log(path: Path, model_version: str | None = None) -> pd.DataFrame:
    """Metrik forward hanya untuk baris yang hasilnya sudah terjadi DAN tercatat tepat waktu (lihat on_time):
    hit rate, log loss, Brier, AUC, plus log loss naif 0,5 dan QLIKE volatilitas."""
    log = read_log(path)
    log = log[pd.to_numeric(log["y_aktual"], errors="coerce").notna()]
    log = log[on_time(log)] if len(log) else log
    if model_version:
        log = log[log["model_version"] == model_version]
    rows = []
    for mv, d in log.groupby("model_version"):
        y, p = d["y_aktual"].astype(float).to_numpy(), d["p_naik"].astype(float).to_numpy()
        r2 = d["r_aktual_pct"].astype(float).to_numpy() ** 2
        s2 = d["sigma_pct"].astype(float).to_numpy() ** 2
        rows.append({"model_version": mv, **oos.metrics(y, p),
                     "log_loss_naif_0_5": float(np.log(2)), "qlike_vol": float(vol.qlike(r2, s2).mean())})
    return pd.DataFrame(rows)
