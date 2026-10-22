# IHSG vs Pasar Global: Korelasi, Lead-Lag, dan Uji Sinyal

Project portfolio data science / quant. Pertanyaan utamanya:

1. Indeks, mata uang, sentimen, dan komoditas global mana yang hubungannya paling kuat dengan IHSG?
2. Apakah hubungan itu stabil, atau berubah saat krisis (2008, 2013, 2015, 2018, 2020, 2022)?
3. Apakah pergerakan pasar yang tutup setelah IHSG (mis. S&P 500) bisa memberi sinyal arah IHSG hari berikutnya, dan apakah sinyal itu masih tersisa setelah harga pembukaan IHSG?

Prinsipnya: **korelasi ≠ prediksi**. Klaim "bisa memberi sinyal" hanya dibuat kalau lolos uji out-of-sample.

> Bukan rekomendasi investasi. Data dari Yahoo Finance (sumber tidak resmi); keterbatasannya ditulis eksplisit di bawah.

## Status

| Phase | Isi | Status |
|---|---|---|
| 1/7 | Setup, data ingestion, data quality | Selesai (run data asli 21 Okt 2026) |
| 2/7 | Alignment, cleaning, returns (harian/mingguan, IHSG dalam USD; close-to-close) | Selesai (21 Okt 2026); USD/IDR lolos uji ulang → driver |
| 3/7 | Korelasi statis (harian vs mingguan, Pearson vs Spearman, IDR vs USD), beta HAC, korelasi asimetris | Selesai (21 Okt 2026); lihat "Temuan" |
| 4/7 | Lead-lag & efek zona waktu (CCF, regresi lead, Granger, premium EIDO; in-sample 2010–2019) | Selesai (21 Okt 2026); lihat "Temuan" |
| 5/7 | Rolling correlation & regime krisis (event study, quantile regression Brent, aset pelindung) | Selesai (23 Okt 2026); lihat "Temuan" |
| 6/7 | Uji sinyal out-of-sample | Belum |
| 7/7 | Ringkasan portfolio | Belum |

## Struktur

```
config/tickers.yaml        # satu-satunya tempat menambah / memindahkan seri
src/                       # semua logika (config, data_loader, quality, preprocess, plots, ...)
notebooks/01-07_*.ipynb    # laporan berjalan; hanya memanggil fungsi dari src/
reports/                   # hasil kecil yang di-commit (CSV, JSON keputusan); target lain di reports/targets/<id>/
scripts/screening.py       # artefak beku: screening Core vs Nanti 2010-2019
tests/                     # pytest dengan data sintetis (tanpa internet)
data/                      # cache download, tidak di-commit
```

## Setup

```bash
conda env create -f environment.yml
conda activate ihsg-corr
python -m ipykernel install --user --name ihsg-corr --display-name "Python 3 (ihsg-corr)"
pytest                     # semua tes harus lulus
jupyter notebook notebooks/01_data_ingestion.ipynb
```

Kalau editable install di `environment.yml` gagal, jalankan `pip install -e .` setelah `conda activate`.
Environment yang sudah ada diperbarui dengan `conda env update -n ihsg-corr -f environment.yml`.

**Mengganti target** (mis. untuk Phase 7) cukup lewat config:
- Ubah `project.target_id` di `config/tickers.yaml` ke id dari blok `targets`, lalu jalankan notebook seperti biasa.
- Output target selain IHSG ditulis ke `reports/targets/<id>/` dan `data/processed/targets/<id>/`.
- Uji ulang USD/IDR tidak dijalankan ulang, karena dikunci terhadap IHSG.
- Sanity check 2008 dilewati kalau target tidak punya angka referensi.

Inisialisasi git:

```bash
git init
git add .
git commit -m "Phase 1: scaffold, config, data loader, data quality"
git branch -M main
git remote add origin git@github.com:matthewphilip-quantlab/ihsg-global-correlation.git
git push -u origin main
```

## Seri yang dipakai

Seri dipilih lewat screening 2010–2019 (`scripts/screening.py`, hasil di `reports/screening_2010_2019.csv`). Urutan screening: kelayakan data → relevansi (korelasi mingguan atau lead t→t+1) → redundansi → informasi tambahan (partial correlation), dengan kuota per channel.

| Seri | Ticker | Sesi vs IHSG | Dasar |
|---|---|---|---|
| Straits Times | ^STI | sinkron | Korelasi mingguan tertinggi (0,53) |
| KLCI Malaysia | ^KLSE | sinkron | Info tambahan +0,33; stabil di 2010–14 dan 2015–19 |
| iShares MSCI EM | EEM | setelah | Info tambahan +0,22; lead terkuat (0,38) |
| S&P 500 | ^GSPC | setelah | Lead 0,34; kandidat utama uji gap pembukaan |
| Nifty 50 | ^NSEI | setelah | Info tambahan +0,14 |
| Hang Seng | ^HSI | sinkron | **Override** (lihat Keputusan) |
| VIX | ^VIX | setelah | Wakil channel risk-off: lead −0,27 |
| Tembaga | HG=F | setelah | Wakil komoditas industri: mingguan 0,28, lead 0,12 |
| Brent | BZ=F | setelah | **Override** (lihat Keputusan) |
| USD/IDR | IDR=X | setelah | **Driver**: lolos uji ulang Phase 2 (mingguan −0,334 pasca-cleaning; lihat poin 4) |

Alat bantu: EIDO (IHSG dalam USD di jam bursa AS; data mulai Mei 2010). Seri ekspansi beserta alasannya ada di blok `nanti:` di `config/tickers.yaml`.

## Keputusan & override (dikunci)

Keputusan di bagian ini dibuat **sebelum** melihat hasil analisis berikutnya, supaya tidak ada *moving goalposts*. Perubahan apa pun harus dicatat di sini beserta alasan dan tanggalnya.

### 1. Periode
- **Screening** dikunci di 1 Jan 2010 – 31 Des 2019. Data di luar periode ini tidak dipakai untuk memilih ulang seri.
- **Analisis** dimulai 1 Jan 2007 (supaya GFC 2008 tercakup) kalau ^JKSE lengkap sejak saat itu. Kalau tidak, fallback ke 1 Jan 2010. Keputusan aktual ditulis otomatis di `reports/phase1/decisions.json`.
- Data 2007–2009 dipakai untuk analisis regime (Phase 5), bukan untuk screening.
- Hari suspend BEI (9–10 Okt 2008) diperlakukan sebagai hari libur, bukan data bolong.
- **Baris basi target diperlakukan sebagai hari libur** (keputusan Phase 2, 21 Okt 2026): Close = Close kemarin **dan** Volume 0. Satu-satunya kasus di ^JKSE adalah 1 Des 2017 (libur Maulid Nabi). Close yang sama dengan hari sebelumnya tetapi volumenya ada (30 Jul 2020) tetap dihitung sebagai hari bursa. Implementasi: `holiday_rows` di `src/preprocess.py`.

### 2. Override Hang Seng
EEM memuat bobot China, tetapi tercampur dengan negara EM lain. HSI dipakai untuk **mengisolasi channel China**, yang menyerap 24,02% ekspor nonmigas Indonesia (2025). Justru karena overlap dengan EEM itulah info tambahan HSI setelah EEM bernilai negatif (−0,10), padahal korelasi mingguannya sendiri 0,47. HSI juga dibutuhkan untuk analisis regime 2015 (crash pasar China dan devaluasi yuan).

### 3. Override Brent
1. **Channel fundamental yang tidak bisa diwakili Tembaga.** Indonesia net importir migas (defisit sekitar US$19,7 miliar di 2025). Harga minyak masuk ke IHSG lewat subsidi energi, neraca berjalan, dan ekspektasi inflasi. Jalur ini berbeda dari tembaga, yang merupakan barometer industri global.
2. **Hipotesis: pengaruh Brent bersifat regime, bukan linier.** Screening memakai korelasi Pearson mingguan 2010–2019, yang lemah terhadap hubungan nonlinier dan episodik. Brent mungkin tidak berpengaruh di minggu normal tetapi sangat berpengaruh saat lonjakan harga. Korelasi lemah ditambah fondasi kuat menjadikannya kandidat untuk diuji di Phase 5 (zoom episode krisis), bukan dibuang.
3. **Tanda korelasinya sendiri menarik.** Korelasi mingguan Brent +0,20 di 2010–2019, padahal secara teori minyak adalah beban bagi Indonesia. Pertanyaan risetnya: apakah tandanya berbalik negatif saat episode lonjakan harga? Hipotesis kerjanya: di minggu normal, pergerakan minyak didominasi permintaan global (minyak dan saham EM naik bersama), sedangkan lonjakan karena pasokan (seperti 2022) bekerja lewat jalur beban. Tembaga dipakai sebagai proxy permintaan untuk memisahkan keduanya.

**Catatan kejujuran:** motivasi override ini sebagian berasal dari pengetahuan tentang lonjakan 2022, yang berada di periode out-of-sample. Karena itu, hasil Brent di Phase 6 diberi label "dipilih dengan pengetahuan periode uji" dan tidak dihitung sebagai bukti out-of-sample yang bersih.

### 4. USD/IDR (aturan dikunci sebelum cleaning)
Screening awal menunjukkan korelasi mingguan −0,19, tetapi data IDR=X memiliki 14 hari dengan |return| > 15% (mustahil untuk USD/IDR di 2010–2019) dan 90 hari return nol. Aturan uji ulang:
- **Cleaning yang diizinkan:** hanya membuang dugaan bad tick, yaitu |log return harian| > 5% yang besoknya dibalik minimal 50% ke arah sebaliknya. Harga pada hari itu dijadikan kosong. Tidak ada filter lain.
- **Ukuran:** |korelasi Pearson| return log mingguan (Jumat–Jumat) USD/IDR vs IHSG, periode 2010–2019, metode sama dengan `scripts/screening.py`.
- **Batas:** |r| ≥ 0,300, dihitung 3 desimal tanpa pembulatan ke atas.
- **Hasil:** lolos → driver (Core). Tidak lolos → **permanen** alat bantu seperti EIDO (tetap dipakai untuk konversi IHSG ke USD).

**Amandemen 1 (21 Okt 2026), dibuat sebelum korelasi pasca-cleaning dihitung.** Disetujui pemilik project.
- **Aturan baru (menggantikan definisi cleaning di atas):** bad tick = |log return harian| > 5% yang dibalik minimal 50% ke arah sebaliknya **dalam ≤ 5 hari bursa**. Semua harga dari hari lonjakan sampai sehari sebelum pembalikan dijadikan kosong. Aturan lama adalah kasus khusus dengan batas 1 hari. Tidak ada filter lain. Aturan ini generik (tanpa tanggal atau nilai tertentu) dan berlaku untuk semua seri FX.
- **Alasan:** data quality Phase 1 menemukan kuotasi salah USD/IDR yang bertahan 2–3 hari bursa (Nov–Des 2013), sehingga lolos dari aturan satu hari. Dua di antaranya jatuh di hari Jumat (29 Nov dan 6 Des 2013), yaitu harga mingguan yang dipakai ukuran di atas. Dua harga itu menghasilkan return mingguan sekitar −19% dan +22% (±20σ), yang menyumbang lebih dari 60% variansi return mingguan USD/IDR 2010–2019. Dengan aturan lama, ukuran ini praktis hanya mencerminkan dua minggu.
- **Pemilihan batas 5 hari:** satu minggu bursa, ditetapkan sebagai satuan alami dan bukan hasil penyetelan. Dry-run cleaning (tanpa menghitung korelasi) memberi hasil identik untuk batas 3 dan 5 hari: 13 hari dibuang, semuanya harga yang jelas salah, dan tidak ada tanggal 2008 yang ikut terbuang. Batas 2 hari justru keliru: harga asli 4–5 Des 2013 terbuang, sedangkan 29 Nov lolos.
- **Yang tidak berubah:** ukuran (Pearson return log mingguan Jumat–Jumat, metode `scripts/screening.py`), periode 2010–2019, ambang 5%, rasio pembalikan 50%, batas |r| ≥ 0,300 tanpa pembulatan ke atas, dan konsekuensi lolos/tidak lolos. Ambang 5% sengaja tidak diturunkan untuk menangkap 21 Okt 2013 (−4,94%), karena itu berarti menyetel parameter berdasarkan satu titik data yang sudah dilihat. Tanggal itu jatuh di hari Senin, jadi tidak memengaruhi ukuran mingguan.
- **Kejujuran:** korelasi sebelum cleaning (−0,19, hasil screening) sudah diketahui. Korelasi pasca-cleaning, baik dengan aturan lama maupun aturan baru, belum dihitung siapa pun saat amandemen ini ditulis.
- **Pelaporan:** hasil yang mengikat memakai aturan baru. |r| dengan aturan lama (batas 1 hari) dilaporkan berdampingan sebagai sensitivitas yang tidak mengikat. Kalau status lolos/gagal keduanya berbeda, hal itu ditulis secara eksplisit.
- **Implementasi:** fungsi cleaning di `src/` dengan tes data sintetis, termasuk kasus run beberapa hari dan kasus batas 2 hari yang keliru sebagai regresi. Dikerjakan di Phase 2 (`src/preprocess.py`, `tests/test_preprocess.py`).

**Hasil uji ulang (Phase 2, 21 Okt 2026; `reports/phase2/usdidr_retest.csv`):**

| Varian | r mingguan 2010–2019 | \|r\| 3 desimal | Minggu | Status |
|---|---|---|---|---|
| Sebelum cleaning | −0,1895 | 0,189 | 507 | tidak lolos |
| Aturan lama (≤ 1 hari), sensitivitas, tidak mengikat | −0,1895 | 0,189 | 507 | tidak lolos |
| **Amandemen 1 (≤ 5 hari), MENGIKAT** | **−0,3340** | **0,334** | 507 | **lolos → driver (Core)** |

- **Keputusan: USD/IDR menjadi driver (Core).** Status di `config/tickers.yaml` diperbarui menjadi `driver`.
- **PERHATIAN: status lolos/gagal berbeda antara aturan lama dan Amandemen 1.** Dengan aturan lama, USD/IDR tidak lolos dan menjadi alat bantu permanen.
  - Seluruh perbedaan berasal dari dua harga Jumat yang salah (29 Nov dan 6 Des 2013). Hanya membuang dua harga itu sudah menghasilkan r = −0,3340.
  - Aturan lama memberi angka yang sama persis dengan data mentah, karena 8 hari yang dibuangnya tidak ada yang menjadi harga mingguan.
- **Catatan kejujuran:** |r| = 0,334 hanya 0,034 di atas batas, kurang dari satu standard error korelasi mingguan (±0,044 untuk ~520 minggu). Statusnya sah menurut aturan yang dikunci, tetapi hubungan USD/IDR–IHSG di level mingguan sebaiknya dibaca sebagai "moderat di ambang batas", bukan kuat.
- **Tanda negatif sesuai teori:** rupiah melemah (USD/IDR naik) bersamaan dengan IHSG turun.

**Hipotesis screening dicabut (Phase 3, 21 Okt 2026).** Screening dengan data mentah memberi kesan "korelasi USD/IDR–IHSG menguat dari 2010–2014 ke 2015–2019" (−0,11 → −0,41; uji Fisher z p < 0,001). Hipotesis itu **dicabut sebagai artefak bad tick 2013**:
- Dengan data bersih (Amandemen 1), korelasinya −0,29 → −0,41. Selisih −0,13 tidak signifikan (p = 0,11).
- Bad tick Okt–Des 2013 jatuh di paruh pertama dan memperlemah korelasi di paruh itu. Sekitar 60% "penguatan" yang terlihat berasal dari bad tick tersebut.
- Detail: `reports/phase3/usdidr_subperiode.csv`.

### 5. Dollar Index → Nanti
Kalah head-to-head dari VIX sebagai wakil channel risk-off / risk-appetite sesi AS: korelasi mingguan −0,11 vs −0,28, lead −0,16 vs −0,27. Tetap ada di blok `nanti:` (tidak dihapus).

### 6. Komoditas ekspor (batubara, nikel, CPO) → Nanti
Tidak ada data harian gratis yang andal. Rencana ekspansi: WHC.AX sebagai proxy batubara harian (screening: mingguan 0,18, 181 hari return nol, jadi kualitasnya perlu dicek dulu), serta seri bulanan IMF via FRED untuk nikel dan CPO. Menambahkannya cukup dengan memindahkan satu blok di `tickers.yaml`.

### 7. Modul gap pembukaan & intraday
Return IHSG dipecah jadi gap pembukaan (close t → open t+1) dan intraday (open → close) **hanya** kalau data Open ^JKSE lolos empat cek: Open kosong ≤ 2%, Open di luar Low–High ≤ 1%, Open = Close kemarin ≤ 5%, dan OHLC datar ≤ 5%. Kalau hanya lolos sejak tahun tertentu (minimal 5 tahun), modul dipakai untuk periode itu saja. Selain itu fallback ke close-to-close, dan keterbatasannya ditulis di bawah.

**Hasil penerapan (21 Okt 2026, disetujui pemilik project):** data Open ^JKSE gagal cek "Open = Close kemarin" (7,6%, batas 5%), dan periode lolos terpanjang hanya 2024–2026 (3 tahun, minimal 5). Karena itu **analisis utama memakai `close_to_close`**. Konsekuensinya, **bagian kedua pertanyaan riset 3 ("apakah sinyal masih tersisa setelah harga pembukaan IHSG") tidak bisa diuji dengan data Yahoo.** Bagian pertama (sinyal pasar sesi "after" di hari t terhadap return close-to-close IHSG di t+1) tetap diuji. Uji gap pembukaan hanya dilakukan sebagai sensitivitas **EKSPLORATIF** di Phase 4 (lihat "Rencana untuk phase berikutnya").

### 8. Kebijakan cleaning (Phase 2)
- **FX:** dugaan bad tick dibuang otomatis dengan aturan di poin 4.
- **Indeks, ETF, komoditas, sentimen:** gerakan besar hanya ditandai lalu diinspeksi manual. Gerakan ini bisa nyata, misalnya Oktober 2008. Pembuangan hanya dilakukan kalau dikonfirmasi salah, dan setiap pembuangan dicatat di sini.
- **Catatan Phase 2 (21 Okt 2026):**
  - Tidak ada pembuangan di indeks, ETF, komoditas, atau sentimen. 27 gerakan besar hanya ditandai (`reports/phase2/large_moves_non_fx.csv`).
  - Di ^JKSE, satu baris basi diperlakukan sebagai hari libur: 1 Des 2017 (libur Maulid Nabi, Close = kemarin, Volume 0).
  - Semua penghapusan, termasuk 13 hari USD/IDR, ada di `reports/phase2/cleaning_log.csv`.

## Rencana teknik per phase (dikunci)

Dikunci 21 Okt 2026, setelah Phase 2 dan **sebelum analisis Phase 3–7 dijalankan**. Perubahan harus dicatat sebagai amandemen bertanggal beserta alasannya, seperti poin 4. Pertanyaan riset yang dirujuk:
- **P1:** seri mana yang hubungannya paling kuat.
- **P2:** apakah hubungan stabil atau berubah saat krisis.
- **P3a:** apakah pasar yang tutup setelah IHSG memberi sinyal arah IHSG hari berikutnya.
- **P3b:** apakah sinyal masih tersisa setelah harga pembukaan. Ini **tidak bisa diuji** dengan data Yahoo (poin 7).

**Aturan umum untuk semua phase**
- **Seleksi** seri, fitur, lag, dan hiperparameter hanya memakai **2010–2019**. Data 2007–2009 dan 2020+ boleh dipakai untuk deskripsi, tetapi tidak untuk memilih.
- **Klaim sinyal** hanya dari **out-of-sample 2020+** (Phase 6). Hasil in-sample di Phase 3–5 disebut "asosiasi" atau "presedensi", bukan sinyal.
- **Kebersihan OOS mulai Phase 4** (keputusan pemilik, 21 Okt 2026, sebelum Phase 4): semua estimasi Phase 4 (lead-lag, Granger, pilihan lag, kointegrasi EIDO, regresi lead) **hanya memakai data s/d 31 Des 2019**. Data 2020+ tidak disentuh sampai Phase 6. Phase 3 yang sudah berjalan (deskriptif, termasuk periode pasca-2020) tidak memilih apa pun dan tidak diubah.
- **Brent** selalu diberi label **"dipilih dengan pengetahuan periode uji"** (poin 3).
- **Data:** return close-to-close dari `data/processed/` (Phase 2). Analisis harian memakai pasangan tanggal yang sama-sama buka. Analisis mingguan memakai W-FRI.
- **Multiple testing:** tabel yang memuat banyak uji (banyak seri × lag) melaporkan p-value terkoreksi Benjamini–Hochberg (FDR 5%) di samping p-value mentah.
- *Keputusan pemilik 21 Okt 2026:* aturan Benjamini–Hochberg dan istilah "asosiasi/presedensi" untuk hasil in-sample **diterima**.

| Phase | Teknik | Pertanyaan | Rincian yang dikunci |
|---|---|---|---|
| 3 | Korelasi statis | P1 | Pearson & Spearman; harian vs mingguan; IHSG dalam IDR vs USD. Utama 2010–2019, plus 2007–2026 sebagai deskripsi. CI 95% via Fisher z. |
| 3 | Regresi faktor mingguan (HAC) | P1 | IHSG mingguan ~ 10 seri Core, plus versi univariat per seri. Newey–West dengan lag = ⌊4(T/100)^(2/9)⌋. Laporkan β, t-HAC, R², adj. R², dan VIF (STI, KLCI, dan EEM saling berkorelasi). |
| 3 | Korelasi asimetris | P2 | Korelasi IHSG–seri di minggu global turun vs naik. "Global" = tanda return mingguan S&P 500; versi sekunder memakai tanda return seri itu sendiri. Dibandingkan dengan korelasi kondisional yang diharapkan di bawah normal bivariat (koreksi bias kondisional Boyer–Gibson–Loretan), karena memotong sampel saja sudah menurunkan korelasi. Uji selisih dengan stationary block bootstrap. |
| 4 | Cross-correlation lead-lag | P3a | Harian, lag −5…+5 hari bursa bersama, 2010–2019. Seri sesi "after" diharapkan muncul di lag +1. |
| 4 | Granger | P3a | **Uji PRESEDENSI PREDIKTIF, bukan sebab-akibat.** VAR bivariat return harian 2010–2019, orde lag dipilih BIC (maks. 5), Wald robust. Dua arah: seri → IHSG dan IHSG → seri. |
| 4 | Sinyal semalam premium/diskon EIDO | P3a | Premium = log EIDO − log IHSG dalam USD (EIDO tutup di sesi AS, setelah IHSG). Uji kointegrasi Engle–Granger dan Johansen di 2010–2019. Kalau terkointegrasi, deviasi dari hubungan jangka panjang dipakai sebagai prediktor return IHSG t+1 (regresi HAC). Kalau tidak, yang dipakai perubahan harian premium, dan hal itu ditulis eksplisit. Catatan: EIDO melacak MSCI Indonesia, bukan IHSG (basis risk), dan timing IDR=X (lihat Keterbatasan data). |
| 5 | Rolling correlation | P2 | Utama: jendela 52 minggu (mingguan). Sekunder: 250 hari bursa (harian). |
| 5 | Event study 7 krisis | P2 | **Satu aturan untuk semua episode:** jendela [−20, +60] hari bursa IHSG di sekitar tanggal jangkar. **Prinsip jangkar:** guncangan global yang menular ke pasar emerging, diambil dari kalender kejadian, bukan dari pergerakan IHSG. Jangkar: 15 Sep 2008 (Lehman), 22 Mei 2013 (taper tantrum), 11 Agu 2015 (devaluasi yuan), 10 Agu 2018 (krisis lira Turki/penularan EM; konsisten dengan 2013 dan 2015), 11 Mar 2020 (WHO: pandemi), 24 Feb 2022 (invasi Rusia ke Ukraina). **Sensitivitas berlabel:** 2018 dijalankan ulang dengan jangkar 5 Feb 2018 (lonjakan VIX). **Okt 2026** (jangkar 16 Okt 2026 diambil dari IHSG sendiri, dan episodenya terpotong oleh akhir data) hanya dilaporkan **deskriptif** dan **dikeluarkan** dari rata-rata atau perbandingan lintas episode. Pembanding "normal" = semua hari di luar ketujuh jendela, 2007–sekarang. |
| 5 | Quantile regression (hipotesis Brent) | P2 | IHSG mingguan ~ Brent + Tembaga (proxy permintaan), τ ∈ {0,05; 0,10; 0,25; 0,50; 0,75; 0,90; 0,95}. Pelengkap: regresi HAC dengan interaksi Brent × 1[return Brent > persentil 90 periode 2010–2019], untuk menguji apakah tanda berbalik saat lonjakan. Berlabel Brent. |
| 5 | Aset pelindung (emas, DXY) | P2 | **Deskriptif saja:** return dan korelasinya dengan IHSG di jendela krisis vs normal. Seri diambil dari blok `nanti:` tanpa dipindah ke Core. |
| 6 | Baseline naif vs logistic regression (arah) | P3a | Target: arah return close-to-close IHSG t+1. Baseline: kelas mayoritas data latih dan persistensi (arah hari t). Logit L2 dengan fitur return hari t dari seri sesi "after"; set fitur dan C ditetapkan dari 2010–2019 (CV deret waktu di dalam data latih) dan dicatat sebelum OOS dijalankan. **Walk-forward:** expanding window sejak 2010, refit bulanan, OOS 2020-01-01 s/d akhir data. Metrik: akurasi, Brier, log loss, AUC. |
| 6 | GARCH-X volatilitas dengan VIX | P3a | GARCH(1,1) return harian IHSG (`arch`) vs GARCH-X dengan perubahan log VIX hari t di persamaan varians. Karena `arch` tidak menyediakan regresor eksogen di persamaan varians, likelihood GARCH-X ditulis sendiri dan divalidasi: koefisien VIX = 0 harus mereproduksi `arch`. Loss: QLIKE dan MSE terhadap return kuadrat. Walk-forward sama seperti logit. |
| 6 | HAR-X volatilitas (pembanding) | P3a | Pembanding yang mudah diverifikasi untuk GARCH-X: regresi OLS volatilitas realisasi IHSG terhadap lag volatilitas harian, mingguan (5 hari), dan bulanan (22 hari), plus perubahan log VIX hari t. Volatilitas realisasi diproksikan dengan \|return harian\| karena tidak ada data intraday. Versi tanpa VIX (HAR) menjadi baseline. Walk-forward dan loss sama seperti GARCH. |
| 6 | Uji Diebold–Mariano | P3a | Logit vs baseline (Brier, log loss), GARCH-X vs GARCH, dan HAR-X vs HAR (QLIKE, MSE), dengan koreksi sampel kecil Harvey–Leybourne–Newbold dan varians HAC. |
| 7 | Ringkasan + target lain | P1–P3a | Pipeline Phase 1–6 dijalankan ulang dengan `project.target_id` = BBCA, BBRI, TLKM, **PTBA**, ASII. Core, aturan, dan periode **sama** (tidak ada seleksi ulang per saham); output di `reports/targets/<id>/`. Fungsinya cek heterogenitas, bukan seleksi baru. *Keputusan pemilik 21 Okt 2026:* PTBA menggantikan ADRO (patahan Nov 2024), dan ANTM tetap di luar (data 2008–2010 kotor). Lihat Keterbatasan data. |

**Phase 6, ditulis eksplisit:** tanpa data Open yang layak, **backtest trading tidak realistis**. Sinyal baru bisa dieksekusi di harga pembukaan, sedangkan gap pembukaan tidak bisa diukur. Karena itu hasil Phase 6 adalah **bukti statistik daya prediksi, bukan klaim return**.

### Spesifikasi Phase 6 (dikunci sebelum Phase 5)

Dikunci oleh pemilik **23 Okt 2026**, setelah Phase 4 (in-sample 2010–2019) dan **sebelum** analisis lead-lag apa pun dijalankan pada data 2020+. Spesifikasi ini **menggantikan** baris Phase 6 di tabel di atas kalau ada perbedaan: baseline, set fitur, dan loss untuk Diebold–Mariano.

- **OOS:** 2020-01-01 s/d data terakhir. Walk-forward dengan expanding window mulai 2010 dan refit bulanan. **Tidak ada tuning hyperparameter yang memakai data OOS.** Kalau regularisasi logit perlu dipilih, pemilihannya hanya dari data latih di setiap titik refit.
- **Target arah:** tanda return close-to-close IHSG(t+1).
- **Model utama:** 4 fitur yang semuanya sudah diketahui setelah IHSG tutup di hari t, ditambah satu kontrol.
  - Fitur: return EEM(t), return S&P 500(t), perubahan log VIX(t), dan **Δpremium EIDO(t)**.
  - Kontrol: return IHSG(t).
- **Model sensitivitas:**
  - Semua seri Core sesi "after" yang Granger-signifikan di Phase 4: EEM, S&P 500, Nifty, VIX, Tembaga, dan Brent (Brent tetap berlabel "dipilih dengan pengetahuan periode uji"). Kontrolnya sama dengan model utama, yaitu IHSG(t).
  - **Rupiah tidak dipakai sebagai fitur** karena tidak mendahului IHSG (Phase 4: justru IHSG yang mendahului rupiah).
  - **z-score spread EIDO versi tren** (keputusan A) boleh diuji sebagai sensitivitas berlabel **"spesifikasi dipilih setelah melihat hasil Phase 4"**. Δpremium tetap fitur utama.
- **Model:** baseline naif (frekuensi hari naik historis di data latih, sebagai probabilitas) vs logistic regression. LightGBM tetap stretch goal.
- **Metrik arah:** hit rate, log loss, Brier, AUC. **Uji Diebold–Mariano vs baseline dengan loss = log loss.**
- **Volatilitas:** GARCH-X dan HAR-X dengan VIX(t) vs GARCH dan HAR tanpa VIX. Metrik QLIKE dan MSE, dengan uji Diebold–Mariano.
- **Peringatan tradeability (ditulis sebelum hasil OOS):**
  - Phase 4 menunjukkan bahwa sebagian besar reaksi IHSG terhadap kabar semalam terjadi di **gap pembukaan**. Karena itu, daya prediksi close-to-close di Phase 6 **TIDAK berarti dapat dieksploitasi**.
  - Kalau data Open 2020+ memungkinkan (setelah hari Open = Close kemarin dibuang), prediksi untuk return **open-to-close** juga dilaporkan sebagai uji **EKSPLORATIF**.

### Aturan Phase 5 (dikunci sebelum Phase 5)
- Analisis Phase 5 boleh memakai seluruh periode (2007–sekarang), tetapi sifatnya **DESKRIPTIF**.
- **Tidak ada temuan Phase 5 yang boleh mengubah Spesifikasi Phase 6 di atas.** Keputusan pemilik, 23 Okt 2026.
- **Rincian pelaksanaan dari instruksi pemilik (23 Okt 2026)**, yang lebih spesifik daripada tabel di atas:
  - Rolling correlation memakai jendela **52 dan 156 minggu**, menggantikan versi sekunder 250 hari harian.
  - Korelasi event study dibandingkan dengan **250 hari bursa sebelum jendela** (tidak tumpang tindih). Pembanding "normal" dari tabel tetap dilaporkan.
  - Quantile regression Brent memakai kontrol **Tembaga dan EEM**.
  - Metrik event study: drawdown maksimum, hari ke titik terendah, dan return kumulatif +60 untuk IHSG, KLCI, STI, dan EEM.
  - Aset pelindung: emas (GC=F), DXY, dan ^TNX dari blok `nanti`, ditambah emas dalam rupiah dan USD/IDR sebagai sudut pandang investor Indonesia.

**Stretch goal (tidak wajib, EKSPLORATIF):** PCA (faktor bersama), VAR/impulse response, HMM untuk regime, uji structural break (Bai–Perron), spillover Diebold–Yilmaz, dan LightGBM + SHAP. Hasilnya tidak dipakai untuk kesimpulan utama. Klaim prediktif dari teknik ini tetap harus lewat OOS 2020+, dan seleksinya tetap hanya dari 2010–2019.

## Rencana untuk phase berikutnya

Dicatat di akhir Phase 1 (21 Okt 2026) supaya tidak ditentukan setelah melihat hasil.

- **Phase 4: sensitivitas gap pembukaan (EKSPLORATIF).**
  - Buang hanya hari dengan Open = Close kemarin (toleransi relatif 1e-6, sama dengan cek Phase 1).
  - Dari sisa harinya, hitung gap pembukaan log(Open t / Close t−1) dan intraday log(Close t / Open t), lalu ulangi uji lead-lag terhadap gap.
  - Hasilnya **tidak dipakai untuk kesimpulan utama** dan selalu diberi label EKSPLORATIF.
  - Catatan bias: hari yang dibuang terkonsentrasi di 2009, 2020, 2022, dan 2023 (bukan acak), sehingga sampel sisanya kurang mewakili periode krisis.
- **Phase 4: harga basi USD/IDR. DIPUTUSKAN** (pemilik, 21 Okt 2026, sebelum Phase 4): harga USD/IDR yang ditandai `flag_stale_quote` **dibuang (NaN) di semua analisis harian mulai Phase 4**. Versi lengkap (tanpa pembuangan) dilaporkan sebagai sensitivitas.
  - Alasannya kualitas data, bukan hasil: nilai 9.612,5 yang berulang bukan harga transaksi.
  - Analisis mingguan (termasuk uji ulang USD/IDR) tidak terpengaruh, karena tidak ada harga yang ditandai menjadi harga mingguan.
- **Phase 5: kandidat regime krisis 16 Okt 2026.** IHSG turun −11,57% dalam sehari (terburuk sejak krisis 1998 menurut pemberitaan; EIDO −16%). Episode ini masuk daftar kandidat regime krisis di samping 2008, 2013, 2015, 2018, 2020, dan 2022.
  - Batas episode ditentukan di Phase 5 dengan aturan yang sama untuk semua episode.
  - Data setelah episode masih sangat pendek saat dicatat, jadi kesimpulan apa pun bersifat awal.

## Temuan

Semua temuan di bawah adalah **asosiasi in-sample (deskriptif), bukan sinyal dan bukan sebab-akibat**. Detail ada di `notebooks/03_static_correlation.ipynb` dan `reports/phase3/`.

**Phase 3, korelasi statis (21 Okt 2026)**
- **Seri paling kuat (mingguan 2010–2019):** Straits Times +0,53, EEM +0,53, KLCI +0,52, Hang Seng +0,47, Nifty +0,46. Urutan antar-periode cukup stabil (Kendall tau 0,82 screening vs penuh), tetapi berubah setelah 2020 (tau 0,56): korelasi Hang Seng turun dari +0,47 ke +0,29, sedangkan S&P 500 naik dari +0,33 ke +0,43.
- **Harian vs mingguan:** untuk seri sesi "after", korelasi harian di hari yang sama hanya sekitar setengah korelasi mingguan (median rasio 0,50; USD/IDR 0,21). Untuk seri "sync" rasionya sekitar 1,0. Reaksi IHSG terhadap pasar yang tutup belakangan baru muncul keesokan harinya, dan itu diuji di Phase 4.
- **Regresi faktor mingguan (HAC, 2010–2019):** R² = 0,43 (pasca-2020 0,36).
  - Dengan seri Core lain dikendalikan: STI +1% → IHSG +0,30%; KLCI +1% → +0,43%; EEM +1% → +0,28%; Nifty +1% → +0,15%; **rupiah melemah 1% → IHSG −0,23%** (CI −0,42% s/d −0,04%).
  - Beta satu faktor rupiah −0,71: sebagian besar asosiasi rupiah tumpang tindih dengan faktor ekuitas regional.
  - Beta S&P 500 berbalik negatif (−0,16) walaupun beta satu faktornya +0,37. Ini efek tumpang tindih dengan EEM (VIF 4,2 dan 5,3), bukan efek terpisah.
  - VIX, Tembaga, dan Brent tidak signifikan setelah faktor ekuitas dikendalikan. Brent berlabel "dipilih dengan pengetahuan periode uji".
- **IHSG dalam USD vs rupiah:** korelasi dengan seri global praktis sama di 2010–2019 (rata-rata |r| +0,01), dan sedikit lebih kuat dalam USD setelah 2020 (+0,05).
- **Asimetri:**
  - Korelasi IHSG lebih tinggi di minggu global turun. IHSG–EEM: +0,52 saat EEM turun vs +0,30 saat naik. IHSG–S&P 500: +0,39 vs +0,22.
  - Memotong sampel saja sudah menurunkan korelasi, jadi angka itu dibandingkan dengan yang diharapkan dari distribusi normal. Asimetri berlebih: EEM +0,23 (CI bootstrap +0,07 s/d +0,37, meyakinkan) dan S&P 500 +0,18 (CI −0,01 s/d +0,36, belum meyakinkan).
  - Uji Fisher z menyatakan keduanya signifikan, tetapi uji itu naif karena mengabaikan efek pemotongan dan autokorelasi.
  - Di 2010–2019 saja, asimetrinya lebih lemah (EEM CI −0,01 s/d +0,37).
- **USD/IDR harian dan kuotasi basi:** membuang 42 return yang ditandai menggeser korelasi harian 2010–2019 dari −0,072 ke −0,091 (Pearson). Pengaruhnya kecil secara absolut, tetapi relatif besar untuk korelasi harian yang memang lemah. Ini masukan untuk keputusan sebelum Phase 4.

**Phase 4, lead-lag & presedensi prediktif (21 Okt 2026; hanya data 2010–2019, kuotasi basi USD/IDR dibuang)**
- **Pola jam perdagangan:**
  - STI, KLCI, dan Hang Seng (sinkron) berkorelasi dengan IHSG di hari yang sama (+0,50 s/d +0,53) dan hampir nol di hari berikutnya.
  - EEM, S&P 500, dan VIX puncaknya di **hari berikutnya** (k = +1: +0,38, +0,34, −0,27).
  - Nifty, Tembaga, dan Brent puncaknya di hari yang sama. Nifty tutup hanya ±1 jam setelah IHSG, sedangkan futures komoditas diperdagangkan hampir 24 jam, termasuk sesi Asia.
- **Regresi lead harian (HAC):**
  - Satu faktor: S&P 500 +1% malam ini ↔ IHSG besok +0,37%; EEM +0,29%; VIX +10% ↔ −0,36%.
  - Multi-faktor (R² 0,127): hanya **EEM** yang tetap kuat (+0,27%). S&P 500 tidak signifikan lagi setelah EEM masuk (VIF 4,7). Tembaga berbalik negatif (−0,07%), yang merupakan efek tumpang tindih.
  - Momentum IHSG sendiri tidak menambah apa-apa.
- **Granger (presedensi prediktif, BH 5%, lag BIC):**
  - EEM, S&P 500, Nifty, VIX, Tembaga, dan Brent → IHSG. Ini diharapkan secara mekanis dari jam perdagangan, dan **bukan sebab-akibat**.
  - Seri sinkron → IHSG: tidak ada.
- **Rupiah tidak berperilaku seperti seri "after", walaupun IDR=X tutup di jam AS.**
  - Korelasi rupiah(t) dengan IHSG(t+1) hanya −0,04 (tidak signifikan).
  - Sebaliknya **IHSG mendahului rupiah**: korelasi IHSG(t−1) vs rupiah(t) −0,14, dan Granger IHSG → rupiah p_bh < 0,001. Konsisten dengan arus dana asing di saham yang dikonversi ke valas setelahnya.
  - Membuang kuotasi basi menggeser korelasi hari yang sama dari −0,071 ke −0,091.
- **EIDO (sinyal semalam):**
  - Return EIDO malam ini saja menjelaskan R² 0,12 dari return IHSG besok. Δpremium (bagian pergerakan EIDO yang belum ada di close IHSG) menjelaskan R² 0,12.
  - Keduanya **menambah informasi di atas S&P 500 dan EEM**: tambahan R² +0,014 (EIDO) dan +0,022 (Δpremium), uji Wald HAC p < 10⁻⁵. Δpremium +1% ↔ IHSG besok +0,14%, setelah S&P 500 dan EEM dikendalikan.
- **Keputusan fitur EIDO (pemilik, 23 Okt 2026):** Δpremium tetap fitur utama sesuai aturan. z-score spread versi tren boleh diuji di Phase 6 hanya sebagai sensitivitas berlabel "spesifikasi dipilih setelah melihat hasil Phase 4".
- **Kointegrasi EIDO vs IHSG dalam USD bergantung spesifikasi:**
  - Aturan utama, Engle–Granger dengan konstanta: p = 0,10, jadi **tidak terkointegrasi** pada α 5%. Sesuai rencana, yang dipakai adalah Δpremium, bukan z-score spread.
  - Engle–Granger dengan tren (p < 10⁻⁵), Johansen, dan ADF premium mentah (p = 0,035) menunjukkan spread stasioner di sekitar tren. Tren itu wajar: harga EIDO di Yahoo sudah disesuaikan dividen, sedangkan IHSG indeks harga. Half-life spread versi tren sekitar 3,3 hari bursa (informasi saja).
- **Gap pembukaan (EKSPLORATIF):**
  - Data Open 2010–2019 hampir bersih (5 dari 2.436 hari dibuang).
  - S&P 500, EEM, dan VIX malam ini berkorelasi ±0,44–0,46 dengan **gap pembukaan** IHSG besok, dan hanya ±0,13–0,25 dengan pergerakan intraday. Sebagian besar "reaksi besok" sudah terjadi di pembukaan.
  - Korelasi intraday kemungkinan dibesar-besarkan oleh Open indeks yang memakai harga kemarin untuk saham yang belum diperdagangkan.
  - Hasil ini tidak dipakai untuk kesimpulan utama.

**Phase 5, regime & krisis (23 Okt 2026; DESKRIPTIF, seluruh periode, tidak mengubah Spesifikasi Phase 6)**
- **Pergeseran pasca-2020: Hang Seng tren, S&P 500 lonjakan sesaat.**
  - Hang Seng (rolling 52 / 156 minggu): rata-rata sebelum 2020 0,45 / 0,49 → 2020–2022 0,48 / 0,52 → **2023–sekarang 0,18 / 0,19**. Penurunannya baru terjadi setelah 2023 dan bertahan, juga di jendela 156 minggu.
  - S&P 500: 0,28 / 0,34 → **2020–2022 0,49 / 0,58** → 2023–sekarang 0,24 / 0,27, kembali ke level lama. Itu guncangan bersama era COVID, bukan pergeseran permanen.
  - Sejak 2023 korelasi IHSG dengan hampir semua pasar regional melemah (STI 0,34, KLCI 0,30, EEM 0,33; rolling 52 minggu).
  - Korelasi Brent bahkan menjadi negatif di jendela terakhir (52 minggu −0,32). Ini deskriptif saja dan berlabel Brent.
- **Event study, median 6 episode (2008, 2013, 2015, 2018 Agu, 2020, 2022):**

  | Seri | Drawdown maks | Hari bursa ke titik terendah | Return +60 hari |
  |---|---|---|---|
  | IHSG | −18,0% | 28 | −5,8% |
  | EEM | −20,1% | 31,5 | −8,0% |
  | STI | −14,3% | 22 | −7,8% |
  | KLCI | −10,0% | 38,5 | −3,4% |

  - IHSG jatuh lebih dalam daripada Singapura dan Malaysia, hampir sedalam EM secara umum.
  - Pemulihannya dalam 60 hari ada di tengah-tengah.
  - Episode terdalam: 2008 (−48,7%) dan 2020 (−33,7%).
  - Sensitivitas jangkar 2018 = 5 Feb: drawdown median sama (−18,0%), tetapi return +60 hari median lebih buruk (−9,7%).
- **Korelasi naik saat krisis** (harian hari yang sama, median setahun sebelum → di dalam jendela): STI +0,37 → +0,66; KLCI +0,41 → +0,59; EEM +0,27 → +0,36. Kenaikan per episode signifikan (Fisher z) di 2015, 2020, dan 2022 untuk STI dan KLCI, tetapi tidak di setiap episode. Manfaat diversifikasi regional menyusut justru saat paling dibutuhkan.
- **Hipotesis Brent tidak didukung data** (*dipilih dengan pengetahuan periode uji*):
  - Di 2010–2019, dengan kontrol Tembaga dan EEM, koefisien Brent praktis nol di semua kuantil (−0,015 s/d +0,002; semua CI memuat nol).
  - Di seluruh periode, koefisien Brent kecil dan **positif**, sedikit lebih besar di ekor bawah (τ 0,25: +0,048, p_bh 0,03; τ 0,1: +0,072, CI memuat nol). Artinya di minggu buruk IHSG, minyak cenderung ikut turun (jalur permintaan), **bukan** naik (jalur beban impor).
  - Interaksi lonjakan Brent (> +4,5% per minggu): tanda **tidak berbalik negatif**. Kemiringan total di minggu lonjakan +0,23 (2010–2019, CI −0,00 s/d +0,46) dan +0,04 (seluruh periode, CI memuat nol).
  - Hipotesis "lonjakan minyak menekan IHSG" tidak terlihat di data mingguan, termasuk 2022.
- **Aset pelindung** (median perubahan dari sebelum jangkar sampai titik terendah IHSG, 6 episode; IHSG −16,4%):
  - **Memegang USD +9,5%** dan **emas dalam rupiah +10,1%**: perlindungan bagi investor rupiah datang terutama dari pelemahan rupiah.
  - Emas dalam USD justru −0,9%. Korelasi mingguannya dengan IHSG **naik** saat krisis (+0,13 → +0,37), karena emas ikut dijual saat likuidasi 2008 dan Mar 2020.
  - Dollar Index +2,6%, dengan korelasi yang makin negatif saat krisis (−0,20 → −0,46): pelindung paling konsisten di antara aset USD.
  - Yield US 10Y +0,08 pp, sehingga tidak melindungi. Di 2013 dan 2022 yield melonjak hampir 1 pp karena guncangannya berupa guncangan suku bunga.
  - **Tidak ada yang berbalik tanda** korelasinya dengan IHSG saat krisis; yang berubah adalah kekuatannya.
- **Okt 2026 (deskriptif, terpotong, tidak masuk ringkasan):** IHSG −14,8% ke titik terendah sehari setelah jangkar, dengan drawdown jendela −22,4%. STI, KLCI, dan EEM nyaris tidak bergerak (−1,7% s/d 0%), jadi episode ini tampak spesifik Indonesia, bukan guncangan global.

## Keterbatasan data

Diperbarui setelah notebook 01 dijalankan dengan data asli (download 21 Okt 2026; detail di `reports/phase1/`).

- Yahoo Finance bukan sumber resmi. Ada bad tick (terutama FX) dan harga basi.
- **Periode analisis mulai 1 Jan 2007** (tanpa fallback): ^JKSE tersedia sejak 2 Jan 2007 dan 2007–2009 lengkap (236–246 hari bursa/tahun).
- **^JKSE kehilangan hari bursa 29 Sep 2008** (hari terakhir sebelum libur Lebaran 30 Sep–3 Okt, tutup ±1.832). Akibatnya return Yahoo pada 6 Okt 2008 (−10,69%) mencakup dua hari bursa; angka resmi −10,03%. Ini satu-satunya FAIL di sanity check 2008 dan sumbernya data Yahoo, bukan angka referensi. Puncak 9 Jan, dasar 28 Okt, penurunan 8 Okt, dan hari suspend 9–10 Okt 2008 semuanya cocok.
- **USD/IDR (IDR=X):**
  - 8 dugaan bad tick sesuai aturan yang dikunci, termasuk dua harga yang kurang satu digit (1 Nov 2010 dan 7 Feb 2012, ±888 dan 892) serta deretan 2013.
  - Harga basi konstan 9.612,5 muncul 34 kali dari Nov 2012 sampai Des 2013, 27 di antaranya hari Senin. Sampai Juli 2013 nilainya dekat kurs asli (selisih 0–4,3%); mulai Okt 2013 selisihnya 14–21%.
  - Aturan awal poin 4 tidak menangkap 29 Nov–3 Des 2013 dan 6–9 Des 2013 (harga salah bertahan 2–3 hari). Dua tanggal ini hari Jumat, jadi masuk return mingguan uji ulang USD/IDR. Ditangani lewat **Amandemen 1** di poin 4 (total 13 hari dibuang).
  - **Sisa yang tetap ada setelah Amandemen 1:** 21 Okt 2013 (−4,94%, di bawah ambang 5%) dan 25 hari basi di Nov 2012–Jul 2013 (21 di antaranya Senin; selisih 0–4,3%).
    - Di Phase 2, ke-25 hari basi itu ditandai otomatis (nilai yang persis berulang ≥ 10 kali, ±0,05%). Tandanya ada di kolom `flag_stale_quote` pada panel harian, yang mencakup 42 return harian pasangan IHSG–USD/IDR. Harga-harga itu tidak dibuang.
    - Tidak ada yang menjadi harga mingguan, jadi uji ulang mingguan tidak terpengaruh. Namun harga-harga itu menghasilkan return harian palsu, dan cara menanganinya untuk analisis harian (Phase 4) diputuskan sebelum Phase 4.
    - 21 Okt 2013 tidak tertangkap penanda ini.
  - Gap 26 hari (1–25 Agu 2008 tidak ada data).
  - Return nol 1,95% dari seluruh hari, terkonsentrasi di 2012–2014 (4,6–8,8% per tahun).
- **IHSG dalam USD (Phase 2)** = ^JKSE / USD/IDR (bersih), dihitung hanya di tanggal yang dua-duanya punya harga.
  - **Keterbatasan timing:** bar harian IDR=X ditutup di jam AS (dini hari WIB), sedangkan IHSG tutup 16:00 WIB. Kurs yang dipakai bukan kurs pada saat IHSG tutup, melainkan kurs beberapa jam setelahnya. Pergerakan rupiah di sesi Eropa/AS ikut masuk ke return harian IHSG dalam USD, sehingga korelasi harian seri ini dengan seri sesi "after" bisa terlalu tinggi.
  - Di level mingguan efek ini jauh lebih kecil. EIDO (diperdagangkan di jam AS) dipakai sebagai pembanding di Phase 4.
- **Return (Phase 2):** log return dihitung hanya di tanggal yang sama-sama buka untuk tiap pasangan IHSG–seri, tanpa forward-fill. Sekitar 1,5–2,3% return harian mencakup jeda lebih dari 4 hari kalender (libur panjang), dan jedanya tercatat di kolom `gap_days`.
- **Gerakan besar di indeks/ETF** yang ditandai dugaan bad tick (EEM 29 Sep 2008, 13 Nov 2008, 16 Mar 2020; HSI 27 Okt 2008; EIDO 12 Mar 2020) jatuh di episode krisis dan tampak nyata (pembalikan tajam di pasar yang panik). Sesuai poin 8, tidak dibuang tanpa konfirmasi.
- **Crash IHSG 16 Okt 2026** (−11,57%, EIDO −16%) terlihat di dua sumber independen, jadi diperlakukan sebagai gerakan nyata. Close Yahoo 5.185,10 sedikit beda dari angka berita (5.199,96); bar hari terakhir bisa belum final saat download. Dicatat sebagai kandidat regime krisis untuk Phase 5.
- **Tanggal mulai:** Nifty 50 baru ada sejak 17 Sep 2007 dan Brent (BZ=F) sejak 30 Jul 2007. Keduanya tetap mencakup 2008. Brent juga punya gap 17 hari hingga 1 Mei 2009.
- **Target saham Phase 7 (dicek 21 Okt 2026, belum dianalisis):**
  - **Keputusan pemilik (21 Okt 2026): ADRO diganti PTBA; ANTM tidak dipakai.** Catatan alasannya di bawah.
  - **PTBA (PTBA.JK), pengganti ADRO:** data sejak 2 Jan 2007, sehingga GFC 2008 tercakup.
    - Split 5:1 tanggal 14 Des 2017 bersih: Close Yahoo sudah disesuaikan, tidak ada lompatan.
    - Dividen tunai: return adjusted Yahoo di ex-date hampir sama dengan total return sebenarnya (selisih ≤ 1,3 pp), **kecuali 26 Jun 2023**. Saat itu dividen Rp1.094 (29,6% harga) dan harga terkunci di batas bawah dua hari (3.700 → 3.150 → 2.680). Return adjusted Yahoo menjadi +20,9%, padahal total return +14,7% (selisih +6,2 pp), lalu −14,9% esok harinya.
    - 13 Sep 2017 (−18,9%, volume ±15× normal) tampak sebagai gerakan nyata.
    - Opsi untuk Phase 7 (belum diputuskan): hitung total return sendiri dari Close tanpa penyesuaian dan dividen, untuk hari ex-date besar.
  - **ADRO (ADRO.JK), tidak dipakai: patahan Nov–Des 2024.**
    - Ticker Yahoo tetap ADRO.JK dan datanya tidak bolong. Nama perusahaan kini PT Alamtri Resources Indonesia Tbk.
    - Pada ex-date 28 Nov 2024 ada dividen spesial Rp1.358,18 per saham (±37% harga) terkait divestasi AADI (Adaro Andalan Indonesia, bisnis batubara termal, listing 5 Des 2024). Saham turun ke batas ARB dua hari berturut-turut (3.670 → 2.760 → 2.080).
    - Penyesuaian dividen Yahoo (auto-adjust) menghasilkan return **+17,7%** di 28 Nov (artefak; total return sebenarnya sekitar +12%), lalu −28,3% di 29 Nov.
    - Yang lebih penting: setelah Des 2024, ADRO secara ekonomi adalah perusahaan lain (tanpa AADI). Eksposurnya ke batubara termal berubah tepat di periode OOS.
    - ADRO juga baru listing 16 Jul 2008, sehingga periode analisisnya fallback ke 2010.
    - Alternatif lain yang sempat dipertimbangkan: ITMG (data sejak 18 Des 2007, fallback 2010), atau ADRO dengan sampel dipotong 26 Nov 2024.
  - **ANTM, tidak dipakai:** data Yahoo 2008–2010 penuh lonjakan ±100% yang langsung berbalik, plus −87% (log) pada 24 Sep 2012. Kebijakan poin 8 hanya menandai gerakan besar non-FX, jadi data ini akan tetap kotor.
  - Saham big cap punya 10–13% hari dengan return tepat nol (tick size, hari sepi), jauh di atas indeks (<0,1%).
- SET (Thailand) dan PSEi (Filipina) tidak tersedia di Yahoo untuk 2010–2019.
- EIDO baru tersedia sejak Mei 2010, jadi tidak ikut analisis 2008.
- Data harian batubara, nikel, dan CPO tidak tersedia gratis.
- **Kualitas data Open ^JKSE: tidak memadai → `return_mode = close_to_close`.**
  - Open = Close kemarin terjadi di 7,6% hari secara keseluruhan (batas 5%). Penyebabnya terkonsentrasi di 2009 (20%), 2020 (45%), 2022 (26%), dan 2023 (57%).
  - Tahun lain hampir bersih. Namun periode bersih terpanjang yang berakhir di tahun terakhir hanya 2024–2026 (3 tahun, minimal 5), jadi mode parsial juga tidak memenuhi syarat.
  - Modul gap pembukaan & intraday tidak dipakai di analisis utama. Bagian kedua pertanyaan riset 3 ("apakah sinyal masih tersisa setelah harga pembukaan") **tidak bisa diuji dengan data Yahoo**; yang tersedia hanya uji sensitivitas eksploratif di Phase 4.
