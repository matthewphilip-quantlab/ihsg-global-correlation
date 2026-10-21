"""
screening.py — Screening kandidat seri untuk project korelasi IHSG
(Phase 1/7, pre-step: menentukan Core vs Nanti berdasarkan angka)

ARTEFAK BEKU: skrip ini disimpan apa adanya sebagai jejak audit pemilihan Core.
Hasilnya ada di reports/screening_2010_2019.csv. Jangan diubah lalu dijalankan
ulang untuk memilih seri baru; daftar seri yang berlaku ada di config/tickers.yaml.

Semua statistik dihitung HANYA dari periode training (default 2010-01-01 s/d
2019-12-31). Tujuannya: pemilihan seri tidak "mengintip" data 2020+ yang nanti
dipakai untuk uji out-of-sample di Phase 6 (menghindari selection / look-ahead bias).

Cara pakai:
    pip install yfinance pandas numpy
    python screening.py
    python screening.py --start 2010-01-01 --train-end 2019-12-31 --out screening_results.csv

Output:
    - tabel di terminal (Core / Nanti / Alat bantu + alasannya)
    - CSV berisi semua kolom statistik
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yfinance as yf

TARGET = "^JKSE"

# ---- Aturan keputusan (ditaruh di atas supaya gampang di-tweak) --------------
MIN_START_LAG_DAYS = 60   # data harus sudah ada paling lambat 60 hari setelah --start
MIN_OBS_PER_YEAR = 200    # normalnya ~240-250 hari bursa/tahun; < 200 = data bolong
MIN_WEEKLY_CORR = 0.30    # |korelasi mingguan| minimal level "moderat"
MIN_LEAD_CORR = 0.10      # |corr X hari t vs IHSG hari t+1|, khusus seri yang tutup setelah IHSG
MAX_REDUNDANCY = 0.85     # |korelasi| dengan seri Core yang sudah terpilih di atas ini = redundan
NOISE_BAND = 0.05         # selisih korelasi < ini dianggap noise (SE korelasi mingguan ~1/sqrt(520) ≈ 0.044)
CORE_CAP = 10             # jumlah maksimal seri Core

# Kuota per channel transmisi: supaya Core menguji tiap jalur pengaruh ke IHSG
# (penularan pasar saham, dolar/suku bunga/risk-off, komoditas), bukan 10 indeks
# Asia yang sebenarnya membawa faktor yang sama. Slot yang tidak terisi
# dialihkan ke kandidat terbaik berikutnya.
CHANNEL = {"index": "ekuitas", "etf": "ekuitas",
           "fx": "makro", "macro": "makro", "sentiment": "makro",
           "commodity": "komoditas"}
CHANNEL_QUOTA = {"ekuitas": 5, "makro": 3, "komoditas": 2}

# session  : "sync"  = tutup sebelum / hampir bersamaan dengan IHSG (~16:00 WIB)
#            "after" = tutup setelah IHSG -> efeknya ke IHSG baru terlihat di hari bursa berikutnya
# transform: "log"   = log return, "diff" = selisih level (untuk yield dalam persen)
# role     : "driver" = kandidat Core/Nanti, "tool" = alat bantu analisis (bukan driver)
# Urutan list = preferensi saat dua seri redundan dan korelasinya beda tipis (dalam NOISE_BAND):
# benchmark yang lebih umum (mis. S&P 500) ditaruh lebih atas.
CANDIDATES = [
    dict(ticker="^GSPC",     name="S&P 500",            category="index",     session="after", transform="log",  role="driver"),
    dict(ticker="^IXIC",     name="Nasdaq Composite",   category="index",     session="after", transform="log",  role="driver"),
    dict(ticker="^HSI",      name="Hang Seng",          category="index",     session="sync",  transform="log",  role="driver"),
    dict(ticker="000001.SS", name="Shanghai Composite", category="index",     session="sync",  transform="log",  role="driver"),
    dict(ticker="^N225",     name="Nikkei 225",         category="index",     session="sync",  transform="log",  role="driver"),
    dict(ticker="^KS11",     name="KOSPI",              category="index",     session="sync",  transform="log",  role="driver"),
    dict(ticker="^TWII",     name="TAIEX",              category="index",     session="sync",  transform="log",  role="driver"),
    dict(ticker="^AXJO",     name="ASX 200",            category="index",     session="sync",  transform="log",  role="driver"),
    dict(ticker="^NSEI",     name="Nifty 50",           category="index",     session="after", transform="log",  role="driver"),
    dict(ticker="^STI",      name="Straits Times",      category="index",     session="sync",  transform="log",  role="driver"),
    dict(ticker="^SET.BK",   name="SET Thailand",       category="index",     session="sync",  transform="log",  role="driver"),
    dict(ticker="^KLSE",     name="KLCI Malaysia",      category="index",     session="sync",  transform="log",  role="driver"),
    dict(ticker="PSEI.PS",   name="PSEi Filipina",      category="index",     session="sync",  transform="log",  role="driver"),
    dict(ticker="^STOXX50E", name="Euro Stoxx 50",      category="index",     session="after", transform="log",  role="driver"),
    dict(ticker="EEM",       name="iShares MSCI EM",    category="etf",       session="after", transform="log",  role="driver"),
    dict(ticker="IDR=X",     name="USD/IDR",            category="fx",        session="after", transform="log",  role="driver"),
    dict(ticker="CNY=X",     name="USD/CNY",            category="fx",        session="after", transform="log",  role="driver"),
    dict(ticker="DX-Y.NYB",  name="Dollar Index",       category="fx",        session="after", transform="log",  role="driver"),
    dict(ticker="^VIX",      name="VIX",                category="sentiment", session="after", transform="log",  role="driver"),
    dict(ticker="^TNX",      name="US Treasury 10Y",    category="macro",     session="after", transform="diff", role="driver"),
    dict(ticker="BZ=F",      name="Brent",              category="commodity", session="after", transform="log",  role="driver"),
    dict(ticker="HG=F",      name="Tembaga",            category="commodity", session="after", transform="log",  role="driver"),
    dict(ticker="GC=F",      name="Emas",               category="commodity", session="after", transform="log",  role="driver"),
    dict(ticker="WHC.AX",    name="Whitehaven (proxy batubara)", category="commodity", session="sync", transform="log", role="driver"),
    # EIDO = IHSG dalam USD yang diperdagangkan di jam AS -> alat untuk Phase 4, bukan driver eksternal
    dict(ticker="EIDO",      name="iShares MSCI Indonesia", category="etf",   session="after", transform="log",  role="tool"),
]


# ---- Data --------------------------------------------------------------------
def download_close(tickers: list[str], start: pd.Timestamp, end_exclusive: pd.Timestamp) -> pd.DataFrame:
    """Download harga close (auto-adjusted) untuk semua ticker sekaligus."""
    raw = yf.download(tickers, start=start, end=end_exclusive, auto_adjust=True,
                      progress=False, threads=True)
    if raw is None or raw.empty:
        raise SystemExit("Download kosong. Cek koneksi / rate limit Yahoo, tunggu sebentar lalu coba lagi.")
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:  # yfinance versi lama dengan satu ticker
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    idx = pd.DatetimeIndex(close.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    close.index = idx.normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close.reindex(columns=tickers)  # ticker yang gagal download -> kolom NaN


def to_returns(s: pd.Series, how: str) -> pd.Series:
    if how == "log":
        return np.log(s.where(s > 0)).diff()
    return s.diff()


# ---- Statistik ---------------------------------------------------------------
def data_quality(s: pd.Series, how: str, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    valid = s.dropna()
    if valid.empty:
        return dict(first_date=pd.NaT, min_obs_year=0, n_spike=np.nan, n_zero=np.nan)
    years = list(range(start.year, end.year + 1))
    tol = pd.Timedelta(days=7)
    full_years = [y for y in years
                  if pd.Timestamp(y, 1, 1) >= start - tol and pd.Timestamp(y, 12, 31) <= end + tol]
    per_year = valid.groupby(valid.index.year).size().reindex(full_years or years, fill_value=0)
    r = to_returns(valid, how).dropna()
    spike_thr = 0.15 if how == "log" else 0.50  # 15%/hari untuk harga, 50 bps/hari untuk yield
    return dict(first_date=valid.index[0], min_obs_year=int(per_year.min()),
                n_spike=int((r.abs() > spike_thr).sum()), n_zero=int((r == 0).sum()))


def daily_pair_stats(close: pd.DataFrame, ticker: str, how: str) -> dict:
    """Korelasi harian. Return dihitung hanya di tanggal yang DUA-DUANYA buka,
    jadi return di sekitar hari libur otomatis mencakup beberapa hari (tanpa forward-fill)."""
    df = close[[TARGET, ticker]].dropna()
    r = pd.DataFrame({"ihsg": to_returns(df[TARGET], "log"),
                      "x": to_returns(df[ticker], how)}).dropna()
    n = len(r)
    if n < 100:
        return dict(n_daily=n, corr_d0=np.nan, corr_lead=np.nan, lead_sig=False)
    corr_d0 = r["x"].corr(r["ihsg"])
    corr_lead = r["x"].corr(r["ihsg"].shift(-1))  # X hari t vs IHSG hari bursa bersama berikutnya
    return dict(n_daily=n, corr_d0=corr_d0, corr_lead=corr_lead,
                lead_sig=bool(abs(corr_lead) > 2 / np.sqrt(n)))


def weekly_returns(close: pd.DataFrame, how_map: dict[str, str]) -> pd.DataFrame:
    wk = close.resample("W-FRI").last()  # harga terakhir yang tersedia tiap minggu
    return pd.DataFrame({t: to_returns(wk[t], how_map[t]) for t in close.columns})


def safe_corr(a: pd.Series, b: pd.Series, min_n: int = 52) -> float:
    pair = pd.concat([a, b], axis=1).dropna()
    return pair.iloc[:, 0].corr(pair.iloc[:, 1]) if len(pair) >= min_n else np.nan


def partial_corr(y: pd.Series, x: pd.Series, Z: pd.DataFrame, min_n: int = 52) -> float:
    """Korelasi y dan x setelah efek linear Z dibuang dari keduanya (residual OLS).
    Menjawab: 'setelah tahu seri-seri Z, berapa informasi BARU yang dibawa x tentang y?'"""
    df = pd.concat([y.rename("_y"), x.rename("_x"), Z], axis=1).dropna()
    if len(df) < min_n:
        return np.nan
    if Z.shape[1] == 0:
        return float(df["_y"].corr(df["_x"]))
    A = np.column_stack([np.ones(len(df)), df[list(Z.columns)].to_numpy()])

    def resid(v: np.ndarray) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ beta

    return float(np.corrcoef(resid(df["_y"].to_numpy()), resid(df["_x"].to_numpy()))[0, 1])


# ---- Keputusan ---------------------------------------------------------------
def decide(res: pd.DataFrame, wret: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    res = res.copy()
    late = res["first_date"].isna() | (res["first_date"] > start + pd.Timedelta(days=MIN_START_LAG_DAYS))
    holey = res["min_obs_year"] < MIN_OBS_PER_YEAR
    res["data_ok"] = ~late & ~holey
    res["relevan"] = (res["corr_w"].abs() >= MIN_WEEKLY_CORR) | (
        (res["session"] == "after") & (res["corr_lead"].abs() >= MIN_LEAD_CORR) & res["lead_sig"]
    )

    decision, reason = {}, {}
    for i, row in res.iterrows():
        if row["role"] == "tool":
            decision[i], reason[i] = "Alat bantu", "dipakai di analisis lead-lag/kurs, bukan driver"
        elif late[i]:
            decision[i] = "Nanti"
            reason[i] = ("data tidak tersedia" if pd.isna(row["first_date"])
                         else f"data baru mulai {row['first_date']:%Y-%m-%d}")
        elif holey[i]:
            decision[i], reason[i] = "Nanti", f"data bolong (min {row['min_obs_year']} obs/tahun)"
        elif not row["relevan"]:
            decision[i] = "Nanti"
            reason[i] = f"hubungan lemah (mingguan {row['corr_w']:+.2f}, lead {row['corr_lead']:+.2f})"

    res["channel"] = res["category"].map(CHANNEL)
    res["partial"] = np.nan
    res["pick_rank"] = np.nan
    tick = res["ticker"]
    passing = res[res["data_ok"] & res["relevan"] & (res["role"] == "driver")]
    order = list(passing["corr_w"].abs().sort_values(ascending=False, kind="stable").index)

    # Langkah 1 — redundansi: dari pasangan dengan |corr| > MAX_REDUNDANCY, simpan satu.
    # Default: yang korelasinya ke IHSG lebih tinggi. Kalau bedanya cuma noise (< NOISE_BAND),
    # pilih yang lebih atas di CANDIDATES (benchmark yang lebih umum).
    kept: list[int] = []
    note: dict[int, str] = {}
    for i in order:
        clash = {j: abs(safe_corr(wret[tick[i]], wret[tick[j]])) for j in kept}
        clash = {j: v for j, v in clash.items() if v > MAX_REDUNDANCY}
        if not clash:
            kept.append(i)
            continue
        j = max(clash, key=clash.get)
        gap = abs(res.at[j, "corr_w"]) - abs(res.at[i, "corr_w"])
        if i < j and gap < NOISE_BAND:
            kept[kept.index(j)] = i
            note[i] = f"; dipilih atas {res.at[j, 'name']} (beda {gap:.2f} = noise)"
            decision[j], reason[j] = "Nanti", f"redundan dengan {res.at[i, 'name']} ({clash[j]:.2f})"
        else:
            decision[i], reason[i] = "Nanti", f"redundan dengan {res.at[j, 'name']} ({clash[j]:.2f})"

    # Langkah 2 — forward selection berdasarkan INFORMASI TAMBAHAN.
    # Tiap langkah pilih seri dengan |partial correlation| terbesar terhadap IHSG,
    # setelah efek seri Core yang sudah terpilih dibuang. Indeks Asia kedua, ketiga, dst.
    # otomatis turun skornya karena faktor Asia sudah tertangkap oleh yang pertama.
    selected: list[int] = []
    pool = kept.copy()
    used = {ch: 0 for ch in CHANNEL_QUOTA}
    while pool and len(selected) < CORE_CAP:
        eligible = [i for i in pool if used[res.at[i, "channel"]] < CHANNEL_QUOTA[res.at[i, "channel"]]]
        spill = not eligible
        if spill:  # kuota semua channel habis -> slot sisa untuk kandidat terbaik
            eligible = pool
        Z = wret[[tick[j] for j in selected]]
        scores = {i: partial_corr(wret[TARGET], wret[tick[i]], Z) for i in eligible}
        best = max(scores, key=lambda i: abs(scores[i]) if np.isfinite(scores[i]) else -1.0)
        selected.append(best)
        pool.remove(best)
        used[res.at[best, "channel"]] += 1
        rank = len(selected)
        res.at[best, "partial"], res.at[best, "pick_rank"] = scores[best], rank
        what = "korelasi terkuat" if rank == 1 else f"info tambahan {scores[best]:+.2f}"
        decision[best] = "Core"
        reason[best] = f"pilihan #{rank}: {what}" + (" [slot sisa]" if spill else "") + note.get(best, "")

    # Sisa kandidat yang lolos tapi tidak terpilih: laporkan info tambahannya setelah seluruh Core.
    Z = wret[[tick[j] for j in selected]]
    for i in pool:
        p = partial_corr(wret[TARGET], wret[tick[i]], Z)
        res.at[i, "partial"] = p
        ch = res.at[i, "channel"]
        full = " (kuota " + ch + " penuh)" if used[ch] >= CHANNEL_QUOTA[ch] else ""
        decision[i], reason[i] = "Nanti", f"info tambahan setelah Core kecil: {p:+.2f}{full}"

    res["keputusan"] = pd.Series(decision)
    res["alasan"] = pd.Series(reason)
    rank_order = {"Core": 0, "Nanti": 1, "Alat bantu": 2}
    res["_o1"] = res["keputusan"].map(rank_order)
    res["_o2"] = res["pick_rank"].fillna(99)
    res["_o3"] = -res["corr_w"].abs().fillna(-1)
    return res.sort_values(["_o1", "_o2", "_o3"]).drop(columns=["_o1", "_o2", "_o3"])


# ---- Main --------------------------------------------------------------------
def main(argv: list[str] | None = None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description="Screening Core vs Nanti untuk korelasi IHSG")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--train-end", default="2019-12-31")
    ap.add_argument("--out", default="screening_results.csv")
    args = ap.parse_args(argv)

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.train_end)
    tickers = [TARGET] + [c["ticker"] for c in CANDIDATES]
    how_map = {TARGET: "log", **{c["ticker"]: c["transform"] for c in CANDIDATES}}

    print(f"Download {len(tickers)} ticker, periode training {start:%Y-%m-%d} s/d {end:%Y-%m-%d} ...")
    close = download_close(tickers, start, end + pd.Timedelta(days=1)).loc[start:end]
    if close[TARGET].dropna().empty:
        raise SystemExit("Data IHSG (^JKSE) gagal didownload. Coba lagi beberapa menit lagi.")

    q_ihsg = data_quality(close[TARGET], "log", start, end)
    print(f"\nIHSG: mulai {q_ihsg['first_date']:%Y-%m-%d}, min {q_ihsg['min_obs_year']} obs/tahun, "
          f"{q_ihsg['n_spike']} hari |return| > 15%, {q_ihsg['n_zero']} hari return tepat nol")
    if q_ihsg["n_zero"] > 20:
        print("  ! Banyak return nol: kemungkinan ada baris basi/hari libur di data Yahoo. Dibersihkan di Phase 2.")

    wret = weekly_returns(close, how_map)
    mid = start + (end - start) / 2

    rows = []
    for c in CANDIDATES:
        t = c["ticker"]
        rows.append({
            **c,
            **data_quality(close[t], c["transform"], start, end),
            **daily_pair_stats(close, t, c["transform"]),
            "n_weekly": int(wret[[TARGET, t]].dropna().shape[0]),
            "corr_w": safe_corr(wret[t], wret[TARGET]),
            "corr_w_h1": safe_corr(wret[t].loc[:mid], wret[TARGET].loc[:mid]),
            "corr_w_h2": safe_corr(wret[t].loc[mid:], wret[TARGET].loc[mid:]),
        })
    res = decide(pd.DataFrame(rows), wret, start)
    res.to_csv(args.out, index=False)

    h1_label = f"{start.year}-{(mid - pd.Timedelta(days=1)).year}"
    h2_label = f"{(mid + pd.Timedelta(days=1)).year}-{end.year}"
    view = pd.DataFrame({
        "Ticker": res["ticker"],
        "Nama": res["name"],
        "Channel": res["channel"].fillna("-"),
        "Sesi": res["session"],
        "Mulai": res["first_date"].dt.strftime("%Y-%m-%d").fillna("-"),
        "Min obs/thn": res["min_obs_year"],
        "Harian (t)": res["corr_d0"].round(2),
        "Lead (t→t+1)": res["corr_lead"].round(2),
        "Mingguan": res["corr_w"].round(2),
        f"Mgg {h1_label}": res["corr_w_h1"].round(2),
        f"Mgg {h2_label}": res["corr_w_h2"].round(2),
        "Info tambahan": res["partial"].round(2),
        "Keputusan": res["keputusan"],
        "Alasan": res["alasan"],
    })
    with pd.option_context("display.width", 260, "display.max_columns", 30, "display.max_colwidth", 60):
        print("\n" + view.to_string(index=False))

    n_core = int((res["keputusan"] == "Core").sum())
    print(f"\nCore: {n_core} seri | Hasil lengkap: {args.out}")
    print(f"Aturan: data mulai <= {MIN_START_LAG_DAYS} hari dari start & >= {MIN_OBS_PER_YEAR} obs/tahun; "
          f"|mingguan| >= {MIN_WEEKLY_CORR} ATAU (sesi 'after' & |lead| >= {MIN_LEAD_CORR} & signifikan); "
          f"tidak redundan (|corr| <= {MAX_REDUNDANCY}); lalu forward selection berdasarkan partial correlation "
          f"dengan kuota channel {CHANNEL_QUOTA}, maksimal {CORE_CAP} seri.")
    return res


if __name__ == "__main__":
    main()
