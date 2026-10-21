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
- Seri ditambah/dipindah hanya lewat `config/tickers.yaml`. Jangan hardcode ticker di kode.
- `data/` tidak di-commit. Output kecil (CSV/JSON) masuk `reports/` dan di-commit. Output notebook (grafik) di-commit.
- Setiap fungsi baru di `src/` diberi tes di `tests/` dengan data sintetis (tanpa internet). Jalankan `pytest` sebelum commit.
- Return dihitung dari log price di tanggal yang sama-sama buka. Tidak boleh forward-fill harga sebelum menghitung return.

## Keputusan yang DIKUNCI
Lihat bagian "Keputusan & override" di README.md. Jangan mengubahnya tanpa persetujuan eksplisit pemilik project.
Yang paling penting:
- Screening window 2010-2019 dikunci; jangan memilih ulang seri dengan data di luar periode itu.
- Aturan uji ulang USD/IDR: |r| mingguan pasca-cleaning >= 0.300, cleaning hanya membuang bad tick.
- Hasil Brent di Phase 6 diberi label "dipilih dengan pengetahuan periode uji".

## Status & langkah berikutnya
- Phase 1/7: kode selesai dan diuji dengan data mock. Langkah berikutnya adalah menjalankan
  `notebooks/01_data_ingestion.ipynb` dengan data asli, membaca `reports/phase1/decisions.json`,
  lalu memperbarui bagian "Keterbatasan data" di README.
- Phase 2/7 (berikutnya):
  - Cleaning sesuai kebijakan di README.
  - Uji ulang USD/IDR.
  - Hapus hari suspend.
  - Hitung return harian dan mingguan, plus IHSG dalam USD.
  - Hitung gap & intraday kalau `return_mode` mengizinkan.
  - Modul baru: `src/preprocess.py` + `notebooks/02_alignment_returns.ipynb`.
