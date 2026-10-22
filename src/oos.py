"""Uji out-of-sample arah IHSG(t+1) (Phase 6). Mengikuti README "Spesifikasi Phase 6 (dikunci sebelum Phase 5)".

Isi modul:
- next_day_target / own_returns / feature_frame : fitur hari t (sudah diketahui setelah IHSG tutup) & target t+1
- timing_audit                                   : cek kebocoran (tanggal target > tanggal fitur)
- walk_forward                                   : expanding window, refit bulanan, baseline naif vs logit
- trend_spread_z                                 : z-score spread EIDO versi tren (fit di data latih saja)
- log_loss / brier / auc / hit_rate / metrics_table / calibration_table
- diebold_mariano                                : DM dengan varians HAC + koreksi Harvey-Leybourne-Newbold
- shuffled_target_placebo / leaked_features / leak_verdict : uji placebo kebocoran
- hit_rate_test                                  : uji tambahan (tidak dipra-registrasi)
- phase6_summary
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from . import analysis as an
from .quality import to_returns

EPS = 1e-15


# ---- Fitur & target ------------------------------------------------------------------------------
def own_returns(close: pd.Series, transform: str = "log") -> pd.Series:
    """Return di hari bursa seri itu sendiri (harga berurutan yang tersedia; tanpa forward-fill)."""
    s = close.dropna()
    return to_returns(s, transform).iloc[1:]


def next_day_target(target_close: pd.Series) -> pd.DataFrame:
    """Untuk tiap hari bursa target t: return target t (kontrol), return t -> t+1 (hari bursa target berikutnya),
    arah naik (1) / tidak (0), dan tanggal realisasinya."""
    s = target_close.dropna()
    r = np.log(s).diff()
    out = pd.DataFrame({"r_target": r, "r_next": r.shift(-1),
                        "target_date": pd.Series(s.index, index=s.index).shift(-1)})
    out["y"] = (out["r_next"] > 0).astype(float).where(out["r_next"].notna())
    return out.iloc[1:]


def feature_frame(close: pd.DataFrame, target_id: str, feature_ids: list[str], transforms: dict[str, str],
                  fx_id: str | None = None, eido_id: str | None = None) -> pd.DataFrame:
    """Baris = hari bursa target t. Fitur = return seri di tanggal kalender yang sama (sesi AS t tutup setelah
    IHSG t, sebelum IHSG t+1 buka); seri yang tidak diperdagangkan di t -> NaN (hari itu tidak diprediksi).
    Kalau eido_id diberi: d_premium(t) = r_EIDO(t) - r_target_USD(t) di tanggal yang target, kurs, & EIDO buka."""
    base = next_day_target(close[target_id])
    feats = {f"r_{c}": own_returns(close[c], transforms.get(c, "log")) for c in feature_ids}
    out = base.join(pd.DataFrame(feats), how="left")
    if eido_id is not None:
        lv = close[[target_id, fx_id, eido_id]].dropna()
        prem = np.log(lv[eido_id]) - np.log(lv[target_id] / lv[fx_id])
        out = out.join(prem.diff().rename("d_premium"), how="left")
        out = out.join(np.log(lv[eido_id]).rename("log_eido"), how="left")
        out = out.join(np.log(lv[target_id] / lv[fx_id]).rename("log_target_usd"), how="left")
    out.index.name = "Date"
    return out


def timing_audit(frame: pd.DataFrame, features: list[str]) -> dict:
    """Cek mekanis: tiap baris memprediksi return yang terealisasi SETELAH tanggal fiturnya."""
    f = frame.dropna(subset=["y"] + features)
    ok = bool((f["target_date"] > f.index).all())
    return {"n_baris": int(len(f)), "target_setelah_fitur": ok,
            "selisih_hari_min": int((f["target_date"] - f.index).dt.days.min()) if len(f) else None}


# ---- Metrik ----------------------------------------------------------------------------------------
def log_loss_obs(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC = statistik Mann-Whitney (peringkat dengan ties dirata-rata)."""
    y = np.asarray(y, dtype=bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = stats.rankdata(p)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"n": int(len(y)), "hit_rate": float(((p > 0.5) == (y == 1)).mean()),
            "log_loss": float(log_loss_obs(y, p).mean()), "brier": float(((p - y) ** 2).mean()),
            "auc": auc(y, p), "frekuensi_naik": float(y.mean())}


def metrics_table(pred: pd.DataFrame, models: list[str], periods: dict[str, tuple] | None = None) -> pd.DataFrame:
    """Metrik per model (kolom p_<model>) dan per periode {nama: (mulai, akhir)}."""
    periods = periods or {"OOS penuh": (None, None)}
    rows = []
    for per, (a, b) in periods.items():
        d = pred.loc[a:b]
        for m in models:
            rows.append({"periode": per, "model": m, **metrics(d["y"].to_numpy(), d[f"p_{m}"].to_numpy())})
    return pd.DataFrame(rows).set_index(["periode", "model"])


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Probabilitas rata-rata vs frekuensi naik aktual per kelompok kuantil prediksi."""
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    g = df.groupby("bin")
    return pd.DataFrame({"p_rata": g["p"].mean(), "frekuensi_naik": g["y"].mean(), "n": g.size()})


def diebold_mariano(loss_model: np.ndarray, loss_base: np.ndarray, h: int = 1) -> dict:
    """DM: d = loss_model - loss_base (negatif = model lebih baik). Varians jangka panjang Newey-West
    (lag = aturan HAC proyek), koreksi sampel kecil Harvey-Leybourne-Newbold, distribusi t(n-1)."""
    d = np.asarray(loss_model, float) - np.asarray(loss_base, float)
    d = d[np.isfinite(d)]
    n = len(d)
    dc = d - d.mean()
    lags = an.hac_lags(n)
    lrv = dc @ dc / n
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * (dc[k:] @ dc[:-k]) / n
    dm = d.mean() / np.sqrt(lrv / n)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(hln), df=n - 1)
    return {"n": n, "selisih_loss_rata": float(d.mean()), "dm_hln": float(hln), "p": float(p),
            "p_satu_sisi_model_lebih_baik": float(stats.t.cdf(hln, df=n - 1))}


# ---- Walk-forward ----------------------------------------------------------------------------------
def trend_spread_z(train: pd.DataFrame, test: pd.DataFrame, origin: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    """z-score spread EIDO versi tren: OLS log_eido ~ const + tren (hari kalender sejak `origin`) + log_target_usd,
    diestimasi di data latih saja; dipakai apa adanya untuk data uji."""
    def design(df):
        t = (df.index - origin).days.to_numpy(dtype=float) / 365.25
        return pd.DataFrame({"const": 1.0, "tren": t, "x": df["log_target_usd"].to_numpy()}, index=df.index)
    tr = train.dropna(subset=["log_eido", "log_target_usd"])
    fit = sm.OLS(tr["log_eido"], design(tr)).fit()
    res = tr["log_eido"] - fit.predict(design(tr))
    mu, sd = res.mean(), res.std()
    z = lambda df: (df["log_eido"] - fit.predict(design(df)) - mu) / sd  # noqa: E731
    return z(train), z(test)


def _fit_logit(X: pd.DataFrame, y: pd.Series):
    """Logit tanpa regularisasi (MLE). Tidak ada hyperparameter, jadi tidak ada tuning. Peringatan statsmodels
    (mis. perfect separation pada uji placebo bocor) diredam; status konvergensi dicatat di hasil walk-forward."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sm.Logit(y, sm.add_constant(X, has_constant="add")).fit(disp=0, maxiter=200)


def walk_forward(frame: pd.DataFrame, models: dict[str, list[str]], train_start: str, oos_start: str,
                 y_col: str = "y", trend_feature: str | None = None) -> pd.DataFrame:
    """Expanding window mulai train_start, refit di awal tiap bulan OOS. Data latih = baris yang targetnya
    sudah TERREALISASI sebelum awal bulan itu (target_date < awal bulan). Prediksi = baris dengan tanggal fitur
    di bulan itu. Baseline = frekuensi naik di data latih. Model yang memuat `trend_feature` mendapat z-score
    spread versi tren yang diestimasi di data latih masing-masing refit."""
    df = frame.loc[train_start:].copy()
    origin = pd.Timestamp(train_start)
    months = pd.date_range(pd.Timestamp(oos_start).to_period("M").to_timestamp(),
                           df.index.max().to_period("M").to_timestamp(), freq="MS")
    out = []
    for m0 in months:
        m1 = m0 + pd.offsets.MonthBegin(1)
        test = df[(df.index >= m0) & (df.index < m1)].dropna(subset=[y_col])
        train = df[(df["target_date"] < m0)].dropna(subset=[y_col])
        if test.empty:
            continue
        if trend_feature is not None:
            ztr, zte = trend_spread_z(train, test, origin)
            train, test = train.assign(**{trend_feature: ztr}), test.assign(**{trend_feature: zte})
        pred = pd.DataFrame({"y": test[y_col], "target_date": test["target_date"],
                             "p_baseline": train[y_col].mean()}, index=test.index)
        for name, feats in models.items():
            tr = train.dropna(subset=feats)
            te = test[feats]
            fit = _fit_logit(tr[feats], tr[y_col])
            ok = te.notna().all(axis=1)
            p = pd.Series(np.nan, index=test.index)
            with np.errstate(over="ignore"):
                p[ok] = fit.predict(sm.add_constant(te[ok], has_constant="add"))
            pred[f"p_{name}"] = p
            pred[f"konvergen_{name}"] = bool(fit.mle_retvals.get("converged", False))
        pred["refit"] = m0
        out.append(pred)
    return pd.concat(out)


def common_rows(pred: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Baris yang semua model punya prediksi (supaya metrik & DM dibandingkan pada hari yang sama)."""
    return pred.dropna(subset=[f"p_{m}" for m in models])


# ---- Placebo kebocoran ---------------------------------------------------------------------------------
def shuffled_target_placebo(frame: pd.DataFrame, feats: list[str], train_start: str, oos_start: str,
                            seeds: int) -> pd.DataFrame:
    """Latih dengan target y DIACAK (urutan diacak, frekuensi tetap), evaluasi pada y asli.
    Tanpa kebocoran, performa OOS harus setara baseline."""
    rows = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        f = frame.copy()
        f["y_acak"] = f["y"].to_numpy()[rng.permutation(len(f))]
        pred = walk_forward(f, {"placebo": feats}, train_start, oos_start, y_col="y_acak")
        pred["y"] = frame.loc[pred.index, "y"]
        pred["p_baseline"] = real_baseline(frame, pred["refit"], train_start)
        pred = common_rows(pred, ["placebo"])
        dm = diebold_mariano(log_loss_obs(pred["y"].to_numpy(), pred["p_placebo"].to_numpy()),
                             log_loss_obs(pred["y"].to_numpy(), pred["p_baseline"].to_numpy()))
        m, b = metrics(pred["y"], pred["p_placebo"]), metrics(pred["y"], pred["p_baseline"])
        rows.append({"seed": s, **m, "log_loss_baseline": b["log_loss"], "selisih_log_loss": m["log_loss"] - b["log_loss"],
                     "dm_hln": dm["dm_hln"], "p_dm": dm["p"]})
    return pd.DataFrame(rows).set_index("seed")


def placebo_summary(res: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Ringkasan uji target acak. Tanpa kebocoran: rata-rata selisih log loss vs baseline >= ~0, AUC tersebar di
    sekitar 0,5 (tanda koefisien acak), dan porsi pengacakan yang 'signifikan lebih baik' tidak jauh di atas alpha/2."""
    better = (res["selisih_log_loss"] < 0) & (res["p_dm"] < alpha)
    return {"n_pengacakan": int(len(res)), "rata_selisih_log_loss": float(res["selisih_log_loss"].mean()),
            "porsi_signifikan_lebih_baik": float(better.mean()), "rata_auc": float(res["auc"].mean()),
            "rata_hit_rate": float(res["hit_rate"].mean())}


def real_baseline(frame: pd.DataFrame, refit: pd.Series, train_start: str) -> pd.Series:
    """Baseline walk-forward dengan y ASLI: frekuensi naik di data latih tiap titik refit."""
    df = frame.loc[train_start:]
    cache = {m0: df.loc[df["target_date"] < m0, "y"].mean() for m0 in refit.unique()}
    return refit.map(cache)


def leaked_features(frame: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """SENGAJA BOCOR: fitur diganti nilainya di hari bursa target berikutnya (t+1), yaitu hari yang sama dengan
    return yang diprediksi. Dipakai hanya untuk membuktikan pipeline mendeteksi kebocoran."""
    f = frame.copy()
    f[feats] = f[feats].shift(-1)
    return f


def hit_rate_test(y: np.ndarray, p_model: np.ndarray, p_base: np.ndarray) -> dict:
    """Uji tanda (McNemar eksak) untuk hit rate model vs baseline di hari yang sama: hanya hari yang hasil tebakannya
    berbeda yang dihitung. TAMBAHAN, tidak dipra-registrasi."""
    a = (np.asarray(p_model) > 0.5) == (np.asarray(y) == 1)
    b = (np.asarray(p_base) > 0.5) == (np.asarray(y) == 1)
    only_m, only_b = int((a & ~b).sum()), int((~a & b).sum())
    res = stats.binomtest(only_m, only_m + only_b, 0.5) if only_m + only_b else None
    return {"hanya_model_benar": only_m, "hanya_baseline_benar": only_b,
            "p": float(res.pvalue) if res else np.nan}


def leak_verdict(leak: pd.DataFrame, honest_hit: float, honest_auc: float, margin: float = 0.03) -> pd.DataFrame:
    """Tandai tiap uji bocor 'melonjak' kalau hit rate ATAU AUC naik > margin di atas model jujur."""
    out = leak.copy()
    out["melonjak"] = (out["hit_rate"] > honest_hit + margin) | (out["auc"] > honest_auc + margin)
    return out


def phase6_summary(main_metrics: pd.DataFrame, dm_main: dict, placebo: dict, leak: pd.DataFrame,
                   vol_tables: dict[str, pd.DataFrame], oc: dict | None, alpha: float = 0.05) -> str:
    """Ringkasan dalam bahasa sederhana. `main_metrics` berindeks model (baseline & utama) pada baris yang sama."""
    b, m = main_metrics.loc["baseline"], main_metrics.loc["utama"]
    win = dm_main["p"] < alpha and dm_main["selisih_loss_rata"] < 0
    L = ["### Ringkasan Phase 6 (OUT-OF-SAMPLE 2020-sekarang, spesifikasi dikunci sebelum Phase 5)", ""]
    L.append(f"- **Arah IHSG besok dari data semalam ({int(m.n):,} hari):** model utama menebak arah benar "
             f"{m.hit_rate:.1%} vs tebakan naif {b.hit_rate:.1%}; log loss {m.log_loss:.4f} vs {b.log_loss:.4f}, "
             f"AUC {m.auc:.3f} vs {b.auc:.3f}. Diebold-Mariano (log loss): p = {dm_main['p']:.3g} -> "
             + ("**lebih baik dari naif secara signifikan**." if win else "**tidak terbukti lebih baik dari naif**."))
    L.append(f"- **Cek kebocoran:** target diacak ({placebo['n_pengacakan']}x) -> rata-rata selisih log loss vs naif "
             f"{placebo['rata_selisih_log_loss']:+.4f}, AUC rata-rata {placebo['rata_auc']:.3f} (setara naif). "
             "Fitur sengaja bocor -> " + "; ".join(
                 f"{i}: hit rate {r.hit_rate:.1%}, AUC {r.auc:.3f} ({'melonjak' if r.melonjak else 'TIDAK melonjak'})"
                 for i, r in leak.iterrows()) + ".")
    for name, tab in vol_tables.items():
        q = tab.loc["QLIKE"]
        better = q["selisih_loss_rata"] < 0 and q["p"] < alpha
        L.append(f"- **{name}:** QLIKE dengan VIX {q.dengan_vix:.4f} vs tanpa {q.tanpa_vix:.4f} (DM p = {q.p:.3g}) -> "
                 + ("VIX **membantu**." if better else "VIX **tidak terbukti membantu**."))
    if oc is not None:
        L.append(f"- **Open-to-close (EKSPLORATIF):** hit rate {oc['hit_rate']:.1%} vs naif {oc['hit_rate_naif']:.1%}, "
                 f"DM p = {oc['p']:.3g}.")
    L.append("- **Tradeability:** daya prediksi close-to-close TIDAK berarti dapat dieksploitasi; sebagian besar reaksi "
             "terhadap kabar semalam terjadi di gap pembukaan (Phase 4), sebelum siapa pun bisa bertransaksi di close.")
    return "\n".join(L)
