# Panduan kerja untuk asisten AI (Claude Code / Codex)

## Konteks
Project portfolio: korelasi IHSG (^JKSE) dengan indeks global, FX, sentimen, dan komoditas.
Ada 7 phase (lihat tabel Status di README.md). Pemilik project adalah data analyst dengan background
data science, mengerti Python/pandas, dan memakai Anaconda + Jupyter Notebook di Mac.

## Cara berkomunikasi
- Jawab dalam Bahasa Indonesia; istilah teknis boleh tetap English.
- Setiap update kode untuk sebuah phase diawali header "Phase X/7" dan 2-3 kalimat tentang langkah
  data science/quant yang dikerjakan dan kenapa penting. Lewati header itu kalau update-nya hanya bug fix.
- Di akhir setiap phase, beri ringkasan insight hasilnya dalam bahasa sederhana.
- Jujur soal batasan: korelasi ≠ prediksi; klaim sinyal hanya lewat uji out-of-sample.

## Aturan kode
- Semua logika di `src/`; notebook (`notebooks/0X_*.ipynb`, berurutan) hanya memanggil fungsi dan menampilkan hasil.
- Seri ditambah/dipindah hanya lewat `config/tickers.yaml`. Jangan hardcode ticker atau id seri di kode/notebook.
- Target dipilih lewat `project.target_id` dari katalog `targets`. Kode memakai `cfg["target"]`, `sanity_ref`,
  `target_usd_id`, dan `output_dir` (`src/config.py`), bukan nama JKSE. Target selain `primary_target` menulis ke
  `reports/targets/<id>/` dan `data/processed/targets/<id>/`. `tests/test_pipeline.py` menjaga Phase 1-2 tetap jalan
  dengan target sintetis.
- `data/` tidak di-commit. Output kecil (CSV/JSON) masuk `reports/` dan di-commit. Output notebook (grafik) di-commit.
- Setiap fungsi baru di `src/` diberi tes di `tests/` dengan data sintetis (tanpa internet). Jalankan `pytest` sebelum commit.
- Return dihitung dari log price di tanggal yang sama-sama buka. Tidak boleh forward-fill harga sebelum menghitung return.

## Keputusan yang DIKUNCI
Lihat bagian "Keputusan & override" di README.md. Jangan mengubahnya tanpa persetujuan eksplisit pemilik project.
Yang paling penting:
- Screening window 2010-2019 dikunci; jangan memilih ulang seri dengan data di luar periode itu.
- Aturan uji ulang USD/IDR: |r| mingguan pasca-cleaning >= 0.300, cleaning hanya membuang bad tick.
- Hasil Brent di Phase 6 diberi label "dipilih dengan pengetahuan periode uji".
- Teknik Phase 3-7 dikunci di README bagian "Rencana teknik per phase (dikunci)": jangan menambah, mengganti, atau
  menyetel ulang teknik/parameter di sana tanpa amandemen bertanggal yang disetujui pemilik. Granger ditulis sebagai
  presedensi prediktif, bukan sebab-akibat. Phase 6 adalah bukti statistik, bukan klaim return (tanpa data Open
  yang layak, backtest trading tidak realistis).

## Status & langkah berikutnya
- Phase 1/7: **selesai** (run data asli 21 Okt 2026; hasil di `reports/phase1/`, catatan di README).
  - Periode analisis mulai 1 Jan 2007 (tanpa fallback).
  - **Amandemen 1** di README poin 4 (opsi B, k = 5): bad tick FX = |log return| > 5% yang dibalik ≥ 50%
    dalam ≤ 5 hari bursa. Ambang, ukuran, dan batas 0.300 tidak berubah. |r| dengan aturan lama (k = 1)
    dilaporkan sebagai sensitivitas yang tidak mengikat. Korelasi pasca-cleaning belum pernah dihitung.
  - Modul Open: fallback **`close_to_close`** diterima untuk analisis utama. Bagian kedua pertanyaan riset 3
    (sinyal setelah pembukaan) tidak bisa diuji dengan data Yahoo; hanya ada sensitivitas EKSPLORATIF di Phase 4.
  - **Harus diputuskan sebelum Phase 4:** sisa harga basi USD/IDR (21 Okt 2013 dan 25 hari Senin bernilai
    ±9.612,5 di Nov 2012–Jul 2013). Tidak memengaruhi uji ulang mingguan, tetapi membuat return harian palsu.
  - Kandidat regime krisis untuk Phase 5: 16 Okt 2026 (IHSG −11,57%).
- Phase 2/7: **selesai** (21 Okt 2026; `src/preprocess.py`, `notebooks/02_alignment_returns.ipynb`, `reports/phase2/`).
  - Uji ulang USD/IDR: |r| = 0.334 (Amandemen 1, mengikat), sehingga USD/IDR menjadi **driver**. Aturan lama memberi 0.189
    (tidak lolos). Status keduanya berbeda, dan itu sudah ditulis eksplisit di README poin 4.
  - Panel di `data/processed/` (tidak di-commit): `prices_clean`, `daily_pair_returns` (dengan `gap_days` dan
    `flag_stale_quote`), `weekly_returns` (W-FRI, termasuk JKSE_USD), `target_usd`, `stale_quote_flags`.
  - Masih terbuka sebelum Phase 4: cara menangani 25 kuotasi basi USD/IDR yang ditandai (42 return harian).
- Rencana teknik Phase 3-7 dikunci (21 Okt 2026), target dibuat configurable, dan `arch` ditambahkan ke environment.
  - Keputusan pemilik (21 Okt 2026, sebelum Phase 3):
    - Benjamini-Hochberg dan istilah "asosiasi/presedensi" untuk hasil in-sample diterima.
    - Event study: prinsip jangkar = guncangan global yang menular ke pasar emerging; 2018 = 10 Agu 2018, dengan
      sensitivitas berlabel 5 Feb 2018. Okt 2026 hanya deskriptif dan DIKELUARKAN dari rata-rata atau perbandingan
      lintas episode.
    - Target Phase 7: BBCA, BBRI, TLKM, PTBA, ASII. ADRO keluar (patahan Nov 2024), ANTM keluar (data 2008-2010
      kotor). Dividen PTBA 26 Jun 2023 membuat return adjusted Yahoo +6,2 pp di atas total return.
    - Phase 6: HAR-X (OLS |return| ~ lag 1/5/22 hari + perubahan log VIX) sebagai pembanding GARCH-X.
  - Label Brent ada di kolom `label` pada `series_table` (dari tickers.yaml); bawa kolom ini ke setiap tabel hasil.
- Phase 3/7 (berikutnya), sesuai rencana yang dikunci:
  - korelasi statis (Pearson/Spearman, harian vs mingguan, IDR vs USD)
  - regresi faktor mingguan HAC
  - korelasi asimetris dengan koreksi bias kondisional
  - Baca panel dari `data/processed/` dan `reports/phase2/decisions.json`.
