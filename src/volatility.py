"""Prediksi volatilitas IHSG out-of-sample: GARCH-X & HAR-X dengan VIX (Phase 6).

GARCH-X (ditulis sendiri karena `arch` tidak menyediakan regresor eksogen di persamaan varians):
    r_t = mu + e_t,  e_t ~ N(0, s2_t)
    s2_t = (omega + alpha e_(t-1)^2 + beta s2_(t-1)) * exp(gamma x_(t-1)),   x = perubahan log VIX
Bentuk perkalian menjamin s2 > 0 walaupun x negatif, dan sama persis dengan GARCH(1,1) saat gamma = 0.
Inisialisasi (backcast dari residual terhadap rata-rata sampel) dan rekursi hari pertama mengikuti `arch`
persis, sehingga log-likelihood di parameter yang sama identik (divalidasi di tes dan notebook).

HAR-X mingguan: log RV_(w+1) ~ log RV_w + log rata-rata RV 4 minggu + log rata-rata RV 13 minggu (+ dlog VIX_w).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

from .oos import diebold_mariano

LOG2PI = np.log(2 * np.pi)


# ---- GARCH-X -------------------------------------------------------------------------------------
def arch_backcast(resids: np.ndarray) -> float:
    """Sama dengan arch: rata-rata tertimbang (0.94^i) kuadrat residual 75 observasi pertama. Seperti arch,
    residualnya dihitung terhadap RATA-RATA SAMPEL (sebelum estimasi), bukan terhadap mu hasil estimasi."""
    tau = min(75, len(resids))
    w = 0.94 ** np.arange(tau)
    return float(np.sum(resids[:tau] ** 2 * w / w.sum()))


def garchx_sigma2(params: np.ndarray, r: np.ndarray, x: np.ndarray, backcast: float | None = None):
    mu, omega, alpha, beta, gamma = params
    e = r - mu
    bc = arch_backcast(r - r.mean()) if backcast is None else backcast
    mult = np.exp(gamma * x) if gamma != 0 else np.ones_like(x)
    s2 = np.empty(len(r))
    s2[0] = omega + (alpha + beta) * bc
    e2 = e * e
    for t in range(1, len(r)):
        s2[t] = (omega + alpha * e2[t - 1] + beta * s2[t - 1]) * mult[t - 1]
    return s2, e


def garchx_loglik(params: np.ndarray, r: np.ndarray, x: np.ndarray) -> float:
    s2, e = garchx_sigma2(params, r, x)
    if np.any(s2 <= 0) or not np.all(np.isfinite(s2)):
        return -np.inf
    return float(-0.5 * np.sum(LOG2PI + np.log(s2) + e * e / s2))


def fit_garchx(r: np.ndarray, x: np.ndarray, gamma_fixed: float | None = None, start: np.ndarray | None = None) -> dict:
    """MLE dengan SLSQP (seperti arch). Kendala: omega > 0, alpha, beta >= 0, alpha + beta < 1."""
    r, x = np.asarray(r, float), np.asarray(x, float)
    v = r.var()
    p0 = np.array([r.mean(), 0.05 * v, 0.08, 0.88, 0.0]) if start is None else np.asarray(start, float).copy()
    free = [0, 1, 2, 3] + ([] if gamma_fixed is not None else [4])
    bounds_all = [(-10 * abs(r.mean()) - 1, 10 * abs(r.mean()) + 1), (1e-8 * v, 10 * v), (0, 1), (0, 1), (-20, 20)]

    def full(theta):
        p = p0.copy()
        p[free] = theta
        if gamma_fixed is not None:
            p[4] = gamma_fixed
        return p

    def nll(theta):
        ll = garchx_loglik(full(theta), r, x)
        return 1e10 if not np.isfinite(ll) else -ll

    cons = [{"type": "ineq", "fun": lambda th: 1 - 1e-6 - th[2] - th[3]}]
    res = minimize(nll, p0[free], method="SLSQP", bounds=[bounds_all[i] for i in free], constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-10})
    p = full(res.x)
    return {"params": p, "loglik": -res.fun, "converged": bool(res.success), "names": ["mu", "omega", "alpha", "beta", "gamma"]}


def validate_against_arch(r: np.ndarray) -> pd.DataFrame:
    """GARCH(1,1) arch vs implementasi sendiri dengan gamma = 0: log-likelihood di parameter arch,
    serta parameter & log-likelihood hasil estimasi sendiri."""
    from arch import arch_model
    r = np.asarray(r, float)
    am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(disp="off")
    pa = np.r_[am.params.to_numpy(), 0.0]
    own_at_arch = garchx_loglik(pa, r, np.zeros_like(r))
    own = fit_garchx(r, np.zeros_like(r), gamma_fixed=0.0)
    names = ["mu", "omega", "alpha", "beta"]
    out = pd.DataFrame({"arch": am.params.to_numpy(), "sendiri": own["params"][:4]}, index=names)
    out.loc["loglik (estimasi masing-masing)"] = [am.loglikelihood, own["loglik"]]
    out.loc["loglik sendiri di parameter arch"] = [am.loglikelihood, own_at_arch]
    out["selisih_abs"] = (out["arch"] - out["sendiri"]).abs()
    out["selisih_relatif"] = out["selisih_abs"] / out["arch"].abs()
    return out


def garch_walk_forward(r: pd.Series, x: pd.Series, train_start: str, oos_start: str, use_x: bool) -> pd.DataFrame:
    """Refit bulanan (expanding sejak train_start). Ramalan varians hari t memakai e_(t-1), s2_(t-1), x_(t-1)
    dengan parameter dari data sebelum bulan itu. Proxy realisasi = r_t^2."""
    r = r.loc[train_start:].astype(float)
    x = x.reindex(r.index).fillna(0.0).astype(float)
    months = pd.date_range(pd.Timestamp(oos_start).to_period("M").to_timestamp(),
                           r.index.max().to_period("M").to_timestamp(), freq="MS")
    start, out = None, []
    for m0 in months:
        m1 = m0 + pd.offsets.MonthBegin(1)
        tr = r.index < m0
        fit = fit_garchx(r[tr].to_numpy(), x[tr].to_numpy(), gamma_fixed=None if use_x else 0.0, start=start)
        start = fit["params"]
        upto = r.index < m1
        s2, _ = garchx_sigma2(fit["params"], r[upto].to_numpy(), x[upto].to_numpy())
        idx = r.index[upto]
        sel = idx >= m0
        out.append(pd.DataFrame({"r2": r[upto][sel].to_numpy() ** 2, "s2": s2[sel], "refit": m0,
                                 "gamma": fit["params"][4]}, index=idx[sel]))
    return pd.concat(out)


def qlike(proxy: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """QLIKE (Patton 2011, bentuk log s2 + proxy/s2): robust terhadap proxy yang berisik, aman untuk proxy = 0."""
    return np.log(s2) + proxy / s2


def mse(proxy: np.ndarray, s2: np.ndarray) -> np.ndarray:
    return (proxy - s2) ** 2


# ---- HAR-X mingguan ---------------------------------------------------------------------------------
def weekly_rv(r: pd.Series) -> pd.DataFrame:
    """RV minggu (W-FRI) = jumlah kuadrat return harian target di minggu itu."""
    g = r.dropna().resample("W-FRI")
    out = pd.DataFrame({"rv": g.apply(lambda s: float((s ** 2).sum())), "n_hari": g.size()})
    return out[out["n_hari"] > 0]


def har_frame(rv: pd.DataFrame, vix_close: pd.Series, weeks: list[int]) -> pd.DataFrame:
    """Fitur di minggu w (log rata-rata RV k minggu terakhir, dlog VIX minggu w) dan target log RV_(w+1)."""
    f = pd.DataFrame(index=rv.index)
    for k in weeks:
        f[f"log_rv_{k}"] = np.log(rv["rv"].rolling(k, min_periods=k).mean())
    vix_w = np.log(vix_close.dropna().resample("W-FRI").last())
    f["dlog_vix"] = vix_w.diff().reindex(f.index)
    f["rv_next"] = rv["rv"].shift(-1)
    f["log_rv_next"] = np.log(f["rv_next"])
    f["target_week"] = pd.Series(rv.index, index=rv.index).shift(-1)
    return f.replace([np.inf, -np.inf], np.nan)


def har_walk_forward(f: pd.DataFrame, feats: list[str], train_start: str, oos_start: str) -> pd.DataFrame:
    """OLS log RV; refit bulanan; data latih = baris yang minggu targetnya berakhir sebelum awal bulan.
    Ramalan varians = exp(prediksi + s^2/2) (koreksi lognormal)."""
    df = f.loc[train_start:].dropna(subset=feats + ["log_rv_next"])
    months = pd.date_range(pd.Timestamp(oos_start).to_period("M").to_timestamp(),
                           df["target_week"].max().to_period("M").to_timestamp(), freq="MS")
    out = []
    for m0 in months:
        m1 = m0 + pd.offsets.MonthBegin(1)
        tr = df[df["target_week"] < m0]
        te = df[(df["target_week"] >= m0) & (df["target_week"] < m1)]
        if te.empty:
            continue
        fit = sm.OLS(tr["log_rv_next"], sm.add_constant(tr[feats])).fit()
        mu = fit.predict(sm.add_constant(te[feats], has_constant="add"))
        out.append(pd.DataFrame({"rv": te["rv_next"], "s2": np.exp(mu + fit.scale / 2), "refit": m0,
                                 "target_week": te["target_week"]}, index=te.index))
    return pd.concat(out)


def compare_losses(pred_x: pd.DataFrame, pred_base: pd.DataFrame, proxy_col: str) -> pd.DataFrame:
    """QLIKE & MSE rata-rata model X vs tanpa X di hari/minggu yang sama, plus uji DM (negatif = X lebih baik)."""
    j = pred_x[[proxy_col, "s2"]].join(pred_base[["s2"]], rsuffix="_base", how="inner")
    rows = []
    for name, fn in (("QLIKE", qlike), ("MSE", mse)):
        lx, lb = fn(j[proxy_col].to_numpy(), j["s2"].to_numpy()), fn(j[proxy_col].to_numpy(), j["s2_base"].to_numpy())
        dm = diebold_mariano(lx, lb)
        rows.append({"loss": name, "dengan_vix": lx.mean(), "tanpa_vix": lb.mean(), **dm})
    return pd.DataFrame(rows).set_index("loss")
