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
| 2/7 | Alignment, cleaning, returns (harian/mingguan, IHSG dalam USD, gap & intraday) | Belum |
| 3/7 | Korelasi statis (harian vs mingguan, Pearson vs Spearman, IDR vs USD) | Belum |
| 4/7 | Lead-lag & efek zona waktu | Belum |
| 5/7 | Rolling correlation & regime krisis | Belum |
| 6/7 | Uji sinyal out-of-sample | Belum |
| 7/7 | Ringkasan portfolio | Belum |

## Struktur

```
config/tickers.yaml        # satu-satunya tempat menambah / memindahkan seri
src/                       # semua logika (config, data_loader, quality, plots, ...)
notebooks/01-07_*.ipynb    # laporan berjalan; hanya memanggil fungsi dari src/
reports/                   # hasil kecil yang di-commit (CSV, JSON keputusan)
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

Inisialisasi git:

```bash
git init
git add .
git commit -m "Phase 1: scaffold, config, data loader, data quality"
git branch -M main
git remote add origin https://github.com/matthewphilip-quantlab/ihsg-global-correlation.git
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
| USD/IDR | IDR=X | setelah | Wajib untuk analisis kurs; status driver diputuskan di Phase 2 (aturan dikunci) |

Alat bantu: EIDO (IHSG dalam USD di jam bursa AS; data mulai Mei 2010). Seri ekspansi beserta alasannya ada di blok `nanti:` di `config/tickers.yaml`.

## Keputusan & override (dikunci)

Keputusan di bagian ini dibuat **sebelum** melihat hasil analisis berikutnya, supaya tidak ada *moving goalposts*. Perubahan apa pun harus dicatat di sini beserta alasan dan tanggalnya.

### 1. Periode
- **Screening** dikunci di 1 Jan 2010 – 31 Des 2019. Data di luar periode ini tidak dipakai untuk memilih ulang seri.
- **Analisis** dimulai 1 Jan 2007 (supaya GFC 2008 tercakup) kalau ^JKSE lengkap sejak saat itu. Kalau tidak, fallback ke 1 Jan 2010. Keputusan aktual ditulis otomatis di `reports/phase1/decisions.json`.
- Data 2007–2009 dipakai untuk analisis regime (Phase 5), bukan untuk screening.
- Hari suspend BEI (9–10 Okt 2008) diperlakukan sebagai hari libur, bukan data bolong.

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
- **Implementasi:** fungsi cleaning di `src/` dengan tes data sintetis, termasuk kasus run beberapa hari dan kasus batas 2 hari yang keliru sebagai regresi. Dikerjakan di Phase 2.

### 5. Dollar Index → Nanti
Kalah head-to-head dari VIX sebagai wakil channel risk-off / risk-appetite sesi AS: korelasi mingguan −0,11 vs −0,28, lead −0,16 vs −0,27. Tetap ada di blok `nanti:` (tidak dihapus).

### 6. Komoditas ekspor (batubara, nikel, CPO) → Nanti
Tidak ada data harian gratis yang andal. Rencana ekspansi: WHC.AX sebagai proxy batubara harian (screening: mingguan 0,18, 181 hari return nol, jadi kualitasnya perlu dicek dulu), serta seri bulanan IMF via FRED untuk nikel dan CPO. Menambahkannya cukup dengan memindahkan satu blok di `tickers.yaml`.

### 7. Modul gap pembukaan & intraday
Return IHSG dipecah jadi gap pembukaan (close t → open t+1) dan intraday (open → close) **hanya** kalau data Open ^JKSE lolos empat cek: Open kosong ≤ 2%, Open di luar Low–High ≤ 1%, Open = Close kemarin ≤ 5%, dan OHLC datar ≤ 5%. Kalau hanya lolos sejak tahun tertentu (minimal 5 tahun), modul dipakai untuk periode itu saja. Selain itu fallback ke close-to-close, dan keterbatasannya ditulis di bawah.

### 8. Kebijakan cleaning (Phase 2)
- **FX:** dugaan bad tick dibuang otomatis dengan aturan di poin 4.
- **Indeks, ETF, komoditas, sentimen:** gerakan besar hanya ditandai lalu diinspeksi manual. Gerakan ini bisa nyata, misalnya Oktober 2008. Pembuangan hanya dilakukan kalau dikonfirmasi salah, dan setiap pembuangan dicatat di sini.

## Keterbatasan data

Diperbarui setelah notebook 01 dijalankan dengan data asli (download 21 Okt 2026; detail di `reports/phase1/`).

- Yahoo Finance bukan sumber resmi. Ada bad tick (terutama FX) dan harga basi.
- **Periode analisis mulai 1 Jan 2007** (tanpa fallback): ^JKSE tersedia sejak 2 Jan 2007 dan 2007–2009 lengkap (236–246 hari bursa/tahun).
- **^JKSE kehilangan hari bursa 29 Sep 2008** (hari terakhir sebelum libur Lebaran 30 Sep–3 Okt, tutup ±1.832). Akibatnya return Yahoo pada 6 Okt 2008 (−10,69%) mencakup dua hari bursa; angka resmi −10,03%. Ini satu-satunya FAIL di sanity check 2008 dan sumbernya data Yahoo, bukan angka referensi. Puncak 9 Jan, dasar 28 Okt, penurunan 8 Okt, dan hari suspend 9–10 Okt 2008 semuanya cocok.
- **USD/IDR (IDR=X):**
  - 8 dugaan bad tick sesuai aturan yang dikunci, termasuk dua harga yang kurang satu digit (1 Nov 2010 dan 7 Feb 2012, ±888 dan 892) serta deretan 2013.
  - Harga basi konstan 9.612,5 muncul 34 kali dari Nov 2012 sampai Des 2013, 27 di antaranya hari Senin. Sampai Juli 2013 nilainya dekat kurs asli (selisih 0–4,3%); mulai Okt 2013 selisihnya 14–21%.
  - Aturan awal poin 4 tidak menangkap 29 Nov–3 Des 2013 dan 6–9 Des 2013 (harga salah bertahan 2–3 hari). Dua tanggal ini hari Jumat, jadi masuk return mingguan uji ulang USD/IDR. Ditangani lewat **Amandemen 1** di poin 4 (total 13 hari dibuang).
  - **Sisa yang tetap ada setelah Amandemen 1:** 21 Okt 2013 (−4,94%, di bawah ambang 5%) dan 25 hari Senin basi di Nov 2012–Jul 2013 (selisih 0–4,3%). Semuanya tidak memengaruhi uji ulang mingguan, tetapi menghasilkan return harian palsu. Cara menanganinya untuk analisis harian (Phase 4) diputuskan sebelum Phase 4.
  - Gap 26 hari (1–25 Agu 2008 tidak ada data).
  - Return nol 1,95% dari seluruh hari, terkonsentrasi di 2012–2014 (4,6–8,8% per tahun).
- **Gerakan besar di indeks/ETF** yang ditandai dugaan bad tick (EEM 29 Sep 2008, 13 Nov 2008, 16 Mar 2020; HSI 27 Okt 2008; EIDO 12 Mar 2020) jatuh di episode krisis dan tampak nyata (pembalikan tajam di pasar yang panik). Sesuai poin 8, tidak dibuang tanpa konfirmasi.
- **Crash IHSG 16 Okt 2026** (−11,57%, EIDO −16%) terlihat di dua sumber independen, jadi diperlakukan sebagai gerakan nyata. Close Yahoo 5.185,10 sedikit beda dari angka berita (5.199,96); bar hari terakhir bisa belum final saat download.
- **Tanggal mulai:** Nifty 50 baru ada sejak 17 Sep 2007 dan Brent (BZ=F) sejak 30 Jul 2007. Keduanya tetap mencakup 2008. Brent juga punya gap 17 hari hingga 1 Mei 2009.
- SET (Thailand) dan PSEi (Filipina) tidak tersedia di Yahoo untuk 2010–2019.
- EIDO baru tersedia sejak Mei 2010, jadi tidak ikut analisis 2008.
- Data harian batubara, nikel, dan CPO tidak tersedia gratis.
- **Kualitas data Open ^JKSE: tidak memadai → `return_mode = close_to_close`.**
  - Open = Close kemarin terjadi di 7,6% hari secara keseluruhan (batas 5%). Penyebabnya terkonsentrasi di 2009 (20%), 2020 (45%), 2022 (26%), dan 2023 (57%).
  - Tahun lain hampir bersih. Namun periode bersih terpanjang yang berakhir di tahun terakhir hanya 2024–2026 (3 tahun, minimal 5), jadi mode parsial juga tidak memenuhi syarat.
  - Modul gap pembukaan & intraday tidak dipakai, dan pertanyaan riset 3 ("apakah sinyal masih tersisa setelah harga pembukaan") hanya bisa diuji secara close-to-close.
