"""Lampiran Teknis: PDF ringkasan notebook 01-07 (BUKAN analisis baru).

Semua angka di teks dibaca dari reports/ (CSV/JSON) saat build; grafik dibuat ulang (200 dpi) dengan fungsi
src/plots.py lewat src/report.py dari data/processed dan reports/.

Pemakaian:
    python scripts/build_notebook_report.py --update-notebooks   # tag sel 'report' & jalankan HANYA sel mandiri baru
    python scripts/build_notebook_report.py                      # bangun reports/Lampiran_Teknis_Notebook.pdf
    python scripts/build_notebook_report.py --render DIR         # + render tiap halaman ke PNG untuk diperiksa
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nbformat  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import analysis, leadlag, oos, preprocess, quality, regime  # noqa: E402
from src import plots as P  # noqa: E402
from src import report as R  # noqa: E402
from src.report import csv, fmt_id, js  # noqa: E402

OUT = ROOT / "reports" / "Lampiran_Teknis_Notebook.pdf"
NB = ROOT / "notebooks"
WIB = timezone(timedelta(hours=7))

# ---- Sel notebook bertag "report" ---------------------------------------------------------------------
TAG_SOURCES = {  # potongan kode sel lama yang hasilnya dipakai di lampiran (hanya diberi tag, tidak dijalankan ulang)
    "01_data_ingestion": ["plots.plot_open_quality(", "plots.plot_jkse_2008(", "q.save_phase1_outputs("],
    "02_alignment_returns": ["plots.plot_fx_cleaning(", "pp.usdidr_retest(", "pp.save_phase2_outputs("],
    "03_static_correlation": ["plots.plot_corr_heatmap(", "plots.plot_asymmetric(", "an.factor_regression(weekly", "an.save_tables("],
    "04_lead_lag": ["plots.plot_ccf_heatmap(", "ll.nested_regressions(", "an.save_tables("],
    "05_regime_crisis": ["plots.plot_rolling_corr(", "plots.plot_event_paths(", "plots.plot_quantile_coefs(", "rg.episode_returns(", "an.save_tables("],
    "06_oos_test": ["plots.plot_calibration(", "oos.shuffled_target_placebo(", "vol.compare_losses(g_x", "an.save_tables("],
    "07_summary_bigcap": ["plots.plot_bigcap_betas(", "bc.comparison_table(rows)", "an.save_tables(rep("],
}
STANDALONE = {  # sel baru, mandiri (membaca reports/ & data/processed), untuk grafik yang belum ada di notebook
    "01_data_ingestion": 'report.show("availability")',
    "02_alignment_returns": 'report.show("return_distribution")',
    "03_static_correlation": 'report.show("beta_forest")',
    "04_lead_lag": 'report.show("eido_premium")',
    "05_regime_crisis": 'report.show("havens")',
    "06_oos_test": ('print(report.save_vol_predictions())   # hitung ulang; disimpan hanya kalau identik dengan reports/phase6\n'
                    'report.show("walkforward", "vol_forecast", "placebo")'),
    "07_summary_bigcap": 'report.show("bigcap_drawdown")',
}
MD_STANDALONE = ("## Lampiran Teknis: grafik tambahan\n\nSel di bawah hanya **menampilkan hasil yang sudah ada**: membaca `reports/` dan "
                 "`data/processed` lewat `src/report.py`, tanpa analisis atau parameter baru. Dipakai oleh "
                 "`scripts/build_notebook_report.py` (Lampiran Teknis PDF).")


def _hash_outputs(cell) -> str:
    return hashlib.sha256(json.dumps(cell.get("outputs", []), sort_keys=True, default=str).encode()).hexdigest()


def update_notebooks() -> None:
    from nbclient import NotebookClient
    for stem, needles in TAG_SOURCES.items():
        path = NB / f"{stem}.ipynb"
        nb = nbformat.read(path, as_version=4)
        before = {c.get("id"): _hash_outputs(c) for c in nb.cells if c.cell_type == "code"}
        for needle in needles:
            hits = [c for c in nb.cells if c.cell_type == "code" and needle in c.source]
            if not hits:
                raise SystemExit(f"{stem}: sel dengan '{needle}' tidak ditemukan")
            for c in hits:
                tags = c.metadata.setdefault("tags", [])
                if "report" not in tags:
                    tags.append("report")
        if not any("report-standalone" in c.metadata.get("tags", []) for c in nb.cells):
            md = nbformat.v4.new_markdown_cell(MD_STANDALONE, metadata={"tags": ["report"]})
            code = nbformat.v4.new_code_cell("from src import report\n" + STANDALONE[stem],
                                             metadata={"tags": ["report", "report-standalone"]})
            nb.cells += [md, code]
        # jalankan HANYA sel mandiri, di kernel terpisah, lalu salin outputnya kembali
        alone = [c for c in nb.cells if "report-standalone" in c.metadata.get("tags", [])]
        tmp = nbformat.v4.new_notebook(cells=[copy.deepcopy(c) for c in alone], metadata=nb.metadata)
        NotebookClient(tmp, timeout=1800, kernel_name="ihsg-corr", resources={"metadata": {"path": str(NB)}}).execute()
        for c, done in zip(alone, tmp.cells):
            c.outputs, c.execution_count = done.outputs, None
        after = {c.get("id"): _hash_outputs(c) for c in nb.cells if c.cell_type == "code" and c.get("id") in before}
        changed = [k for k in before if before[k] != after.get(k)]
        if changed:
            raise SystemExit(f"{stem}: output sel lama berubah {changed}; notebook TIDAK ditulis")
        nbformat.write(nb, path)
        print(f"{stem}: tag 'report' pada {sum('report' in c.metadata.get('tags', []) for c in nb.cells)} sel; "
              f"{len(alone)} sel mandiri dijalankan; output sel lama tidak berubah")


# ---- PDF ------------------------------------------------------------------------------------------------
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Flowable, Frame, Image, KeepTogether,  # noqa: E402
                                NextPageTemplate, PageBreak, PageTemplate, Paragraph, Preformatted, Spacer, Table,
                                TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents  # noqa: E402

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, file in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"),
                   ("DV-M", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")

BLUE, ORANGE, AQUA, YELLOW = (colors.HexColor(c) for c in P.SERIES)
INK, INK2, GRID = colors.HexColor(P.INK), colors.HexColor(P.INK_2), colors.HexColor(P.GRID)
BLUE_BG, ORANGE_BG, CODE_BG = colors.HexColor("#e9f1fb"), colors.HexColor("#fdeee6"), colors.HexColor("#f5f5f2")
W = A4[0] - 36 * mm

S = {
    "body": ParagraphStyle("body", fontName="DV", fontSize=9, leading=12.6, textColor=INK, alignment=TA_LEFT, spaceAfter=4),
    "small": ParagraphStyle("small", fontName="DV", fontSize=8, leading=11, textColor=INK),
    "cap": ParagraphStyle("cap", fontName="DV", fontSize=8, leading=10.6, textColor=INK2, spaceBefore=2, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=10.5, leading=14, textColor=BLUE, spaceBefore=8, spaceAfter=4),
    "box_t": ParagraphStyle("box_t", fontName="DV-B", fontSize=9, leading=12, textColor=INK, spaceAfter=3),
    "box": ParagraphStyle("box", fontName="DV", fontSize=8.3, leading=11.2, textColor=INK, leftIndent=8, bulletIndent=0),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=7.6, leading=9.6, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DV-B", fontSize=7.6, leading=9.6, textColor=colors.white),
    "code": ParagraphStyle("code", fontName="DV-M", fontSize=6.8, leading=8.4, textColor=INK, backColor=CODE_BG,
                           borderPadding=(5, 5, 5, 5), leftIndent=5, rightIndent=5, spaceBefore=4, spaceAfter=10),
    "title": ParagraphStyle("title", fontName="DV-B", fontSize=24, leading=30, textColor=INK),
    "sub": ParagraphStyle("sub", fontName="DV", fontSize=12, leading=17, textColor=INK2),
    "toc1": ParagraphStyle("toc1", fontName="DV", fontSize=9.5, leading=15, textColor=INK),
}


class SectionBand(Flowable):
    """Pita judul bagian berwarna; juga dicatat ke daftar isi & bookmark PDF."""

    def __init__(self, title: str, kicker: str = "", color=BLUE):
        super().__init__()
        self.title, self.kicker, self.color = title, kicker, color
        self.height = 17 * mm

    def wrap(self, aw, ah):
        self.width = aw
        return aw, self.height

    def draw(self):
        c = self.canv
        c.setFillColor(self.color)
        c.roundRect(0, 0, self.width, self.height, 3, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("DV", 8)
        c.drawString(5 * mm, self.height - 6 * mm, self.kicker)
        c.setFont("DV-B", 14)
        c.drawString(5 * mm, 4.2 * mm, self.title)


class Doc(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm,
                         bottomMargin=18 * mm, title="Lampiran Teknis: IHSG vs Pasar Global",
                         author="ihsg-global-correlation", subject="Ringkasan notebook 01-07")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate("cover", [frame], onPage=self._cover),
                               PageTemplate("body", [frame], onPage=self._body)])

    def _cover(self, c, doc):
        c.saveState()
        c.setFillColor(BLUE)
        c.rect(0, A4[1] - 12 * mm, A4[0], 12 * mm, stroke=0, fill=1)
        c.restoreState()

    def _body(self, c, doc):
        c.saveState()
        c.setStrokeColor(GRID)
        c.setLineWidth(0.6)
        c.line(18 * mm, A4[1] - 12 * mm, A4[0] - 18 * mm, A4[1] - 12 * mm)
        c.line(18 * mm, 12 * mm, A4[0] - 18 * mm, 12 * mm)
        c.setFont("DV", 7.5)
        c.setFillColor(INK2)
        c.drawString(18 * mm, A4[1] - 10 * mm, "Lampiran Teknis · IHSG vs Pasar Global: Korelasi, Lead-Lag, dan Uji Sinyal")
        c.drawString(18 * mm, 8 * mm, "Ringkasan notebook 01-07 · bukan analisis baru · bukan rekomendasi investasi")
        c.drawRightString(A4[0] - 18 * mm, 8 * mm, f"hlm. {doc.page}")
        c.restoreState()

    def afterFlowable(self, f):
        if isinstance(f, SectionBand):
            text = f"{f.kicker} · {f.title}" if f.kicker else f.title
            key = f"s{self.seq.nextf('sec')}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=0)
            self.notify("TOCEntry", (0, text, self.page, key))


# ---- Elemen ----------------------------------------------------------------------------------------------
FIG_DIR = Path(tempfile.mkdtemp(prefix="lampiran_"))
_fig_no: dict[int, int] = {}


def figure(phase: int, key: str, caption: str, how: str, max_h: float = 105 * mm) -> KeepTogether:
    fig = R.figure(key)
    path = FIG_DIR / f"{phase}_{key}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _fig_no[phase] = _fig_no.get(phase, 0) + 1
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    w = min(W, max_h / ratio)
    img.drawWidth, img.drawHeight = w, w * ratio
    cap = Paragraph(f"<b>Gambar {phase}.{_fig_no[phase]} · {caption}</b><br/><i>Cara membaca:</i> {how}", S["cap"])
    return KeepTogether([img, cap])


def para(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def table(rows: list[list], widths: list[float], header_color=BLUE, title: str | None = None) -> list:
    data = [[Paragraph(str(c), S["cellb"] if i == 0 else S["cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), header_color), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("LINEBELOW", (0, -1), (-1, -1), 0.6, GRID), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5), ("LEFTPADDING", (0, 0), (-1, -1), 4)]
    st += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f7f7f5")) for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle(st))
    out = [para(title, "h2")] if title else []
    return [KeepTogether(out + [t, Spacer(1, 6)])]


def pipeline(inp: str, proc: str, out: str) -> Table:
    cells = [[Paragraph(f"<b>Input</b><br/>{inp}", S["small"]), Paragraph("→", S["box_t"]),
              Paragraph(f"<b>Proses</b><br/>{proc}", S["small"]), Paragraph("→", S["box_t"]),
              Paragraph(f"<b>Output</b><br/>{out}", S["small"])]]
    t = Table(cells, colWidths=[0.29 * W, 0.04 * W, 0.34 * W, 0.04 * W, 0.29 * W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), BLUE_BG), ("BACKGROUND", (2, 0), (2, 0), BLUE_BG),
                           ("BACKGROUND", (4, 0), (4, 0), BLUE_BG), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("ALIGN", (1, 0), (1, 0), "CENTER"), ("ALIGN", (3, 0), (3, 0), "CENTER"),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return t


def boxes(insights: list[str], rules: list[str]) -> Table:
    def col(title, items, bg):
        return [Paragraph(title, S["box_t"])] + [Paragraph(i, S["box"], bulletText="•") for i in items]
    t = Table([[col("Insight", insights, BLUE_BG), col("Keputusan / aturan yang dikunci", rules, ORANGE_BG)]],
              colWidths=[0.5 * W - 3, 0.5 * W - 3])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), BLUE_BG), ("BACKGROUND", (1, 0), (1, 0), ORANGE_BG),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 7),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("LINEBEFORE", (1, 0), (1, 0), 6, colors.white)]))
    return t


def code(func, where: str, max_lines: int = 15) -> list:
    return [KeepTogether([para(f"<b>Cara kerja</b> · <font name='DV-M'>{func.__name__}</font> ({where})", "h2"),
                          Preformatted(R.snippet(func, max_lines), S["code"], maxLineLength=104, newLineChars="    ")])]


def f2(x, nd=2, sign=False, pct=False):
    return fmt_id(float(x), nd, sign, pct)


def hr(rate: float, n: float) -> float:
    """Hit rate dari jumlah hari benar (round(rate x n) / n): menghindari pembulatan ganda dari CSV 4 desimal."""
    return round(float(rate) * float(n)) / float(n)


BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


def idate(ts, time: bool = False) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts.day:02d} {BULAN[ts.month - 1]} {ts.year}" + (f" {ts:%H:%M}" if time else "")


def p_txt(p: float) -> str:
    return "&lt; 0,001" if p < 0.001 else fmt_id(p, 3)


# ---- Isi ---------------------------------------------------------------------------------------------------
def build(path: Path = OUT) -> tuple[Path, pd.DataFrame]:
    N = {k: v.replace("&", "&amp;") for k, v in R.names().items()}   # nama seri aman untuk markup Paragraph
    cfg = R._cfg()
    checks: list[tuple[str, str]] = []
    d1, d2 = js("phase1/decisions.json"), js("phase2/decisions.json")
    story: list = [NextPageTemplate("body")]

    # ---------- Sampul ----------
    prices, _ = R.processed()
    story = [Spacer(1, 45 * mm), para("Lampiran Teknis", "sub"),
             para("IHSG vs Pasar Global:<br/>Korelasi, Lead-Lag, dan Uji Sinyal", "title"), Spacer(1, 6 * mm),
             para("Ringkasan tujuh notebook (01-07): poin penting, visualisasi, dan hasil kunci per phase.", "sub"),
             Spacer(1, 30 * mm),
             para(f"Data: Yahoo Finance, {idate(d1['analysis_start'])} – {idate(prices.index.max())} · "
                  f"Repo: matthewphilip-quantlab/ihsg-global-correlation · Dibuat {idate(datetime.now(WIB).replace(tzinfo=None), True)} WIB", "small"),
             Spacer(1, 4 * mm),
             para("Dokumen ini <b>tidak berisi analisis baru</b>. Semua angka dibaca dari <font name='DV-M'>reports/</font> "
                  "saat dokumen dibangun, dan grafik dibuat ulang dengan fungsi yang sama di "
                  "<font name='DV-M'>src/plots.py</font>. Bukan rekomendasi investasi.", "small"),
             NextPageTemplate("body"), PageBreak()]

    # ---------- Daftar isi + ringkasan ----------
    toc = TableOfContents()
    toc.levelStyles = [S["toc1"]]
    story += [SectionBand("Daftar isi & ringkasan", "Lampiran Teknis"), Spacer(1, 4), toc, Spacer(1, 6)]
    core = [s["id"] for s in cfg["core"]]
    m6 = csv("phase6/arah_metrik_utama.csv", index_col=0)
    dm6 = csv("phase6/arah_dm.csv", index_col=0)
    g6 = csv("phase6/vol_garch_vs_garchx.csv", index_col=0)
    em = csv("phase5/event_study_median.csv", index_col=0)
    kor = csv("phase3/korelasi.csv")
    wk = kor[(kor["frekuensi"] == "mingguan") & (kor["metode"] == "pearson") & (kor["periode"] == "screening")].set_index("id")
    top3 = wk["r"].abs().sort_values(ascending=False).index[:3]
    top_list = ", ".join(N[i] + " " + f2(wk.loc[i, "r"], 2, True) for i in top3)
    reg = csv("phase3/regresi_faktor_screening.csv", index_col=0)
    one = csv("phase3/beta_satu_faktor_screening.csv", index_col=0)
    hv = R.haven_medians()
    fx = cfg["usdidr_retest"]["id"]
    story += [para("Pertanyaan riset", "h2"),
              para("<b>P1</b> Seri global mana yang paling terkait dengan IHSG? · <b>P2</b> Apakah hubungan itu stabil atau "
                   "berubah saat krisis? · <b>P3</b> Apakah pasar yang tutup setelah Jakarta memberi sinyal arah IHSG "
                   "besok (dan apakah masih tersisa setelah harga pembukaan)?"),
              para("Data", "h2"),
              para(f"Target ^JKSE; 10 seri Core ({', '.join(N[i] for i in core)}); alat bantu EIDO. Seleksi seri hanya dari "
                   f"{cfg['project']['screening_window'][0][:4]}–{cfg['project']['screening_window'][1][:4]}; analisis "
                   f"{d1['analysis_start'][:4]}–{prices.index.max():%Y}; uji out-of-sample {cfg['oos']['oos_start'][:4]}–"
                   f"{prices.index.max():%Y}. Sumber: Yahoo Finance (tidak resmi; keterbatasan dicatat per phase).")]
    flow = [["01 · Data quality", "02 · Cleaning & returns", "03 · Korelasi & beta", "04 · Lead-lag",
             "05 · Regime & krisis", "06 · Uji OOS", "07 · Big cap & ringkasan"]]
    ft = Table([[Paragraph(x, S["cellb"]) for x in flow[0]]], colWidths=[W / 7] * 7)
    ft.setStyle(TableStyle([("BACKGROUND", (i, 0), (i, 0), [BLUE, BLUE, BLUE, BLUE, BLUE, ORANGE, AQUA][i]) for i in range(7)] +
                           [("LINEAFTER", (0, 0), (-2, 0), 2, colors.white), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [para("Alur 7 phase", "h2"), ft, Spacer(1, 4)]
    findings = [
        f"<b>Pasar tetangga & EM dominan:</b> korelasi mingguan 2010–2019 tertinggi {top_list}; "
        f"model 10 faktor R² {f2(js('phase3/summary.json')['model_regresi']['screening']['r2'])}.",
        f"<b>Rupiah:</b> melemah 1% ↔ IHSG {f2(one.loc[fx, 'beta_1f'], 2, True)}% (satu faktor), "
        f"{f2(reg.loc[fx, 'beta'], 2, True)}% setelah ekuitas regional dikendalikan; secara harian rupiah mengikuti IHSG.",
        f"<b>Krisis:</b> median drawdown IHSG {f2(em.loc['JKSE', 'max_drawdown'], 1, pct=True)} vs STI "
        f"{f2(em.loc['STI', 'max_drawdown'], 1, pct=True)} dan KLCI {f2(em.loc['KLCI', 'max_drawdown'], 1, pct=True)}; "
        f"pelindung: USD {f2(hv[fx], 1, True, pct=True)}, emas dalam rupiah {f2(hv['GOLD_IDR'], 1, True, pct=True)}.",
        f"<b>Arah IHSG besok (OOS): tidak terbukti</b> lebih baik dari naif — hit rate {f2(hr(m6.loc['utama', 'hit_rate'], m6.loc['utama', 'n']), 1, pct=True)} "
        f"vs {f2(hr(m6.loc['baseline', 'hit_rate'], m6.loc['baseline', 'n']), 1, pct=True)}, tetapi DM log loss p = {f2(dm6.loc['utama', 'p'], 2)}.",
        f"<b>Volatilitas (OOS): VIX terbukti membantu</b> — GARCH-X QLIKE {f2(g6.loc['QLIKE', 'dengan_vix'], 3)} vs "
        f"{f2(g6.loc['QLIKE', 'tanpa_vix'], 3)}, DM p = {f2(g6.loc['QLIKE', 'p'], 3)}.",
        "<b>Hasil negatif yang dicatat:</b> hipotesis Brent tidak didukung; hipotesis \"korelasi rupiah menguat\" dicabut "
        "sebagai artefak bad tick 2013.",
    ]
    checks += [("korelasi mingguan teratas", f2(wk.loc[top3[0], "r"], 2)),
               ("beta rupiah satu faktor", f2(one.loc[fx, "beta_1f"], 2)), ("beta rupiah multi", f2(reg.loc[fx, "beta"], 2)),
               ("drawdown IHSG", f2(em.loc["JKSE", "max_drawdown"], 1, pct=True)),
               ("drawdown STI", f2(em.loc["STI", "max_drawdown"], 1, pct=True)),
               ("pelindung USD", f2(hv[fx], 1, True, pct=True)), ("pelindung emas IDR", f2(hv["GOLD_IDR"], 1, True, pct=True)),
               ("hit rate utama", f2(hr(m6.loc["utama", "hit_rate"], m6.loc["utama", "n"]), 1, pct=True)),
               ("hit rate naif", f2(hr(m6.loc["baseline", "hit_rate"], m6.loc["baseline", "n"]), 1, pct=True)),
               ("DM arah p", f2(dm6.loc["utama", "p"], 2)), ("DM GARCH-X p", f2(g6.loc["QLIKE", "p"], 3))]
    story += [para("Temuan utama", "h2")] + [Paragraph(f, S["box"], bulletText="•") for f in findings] + [PageBreak()]

    # ---------- Phase 1 ----------
    story += [SectionBand("Data ingestion & kualitas data", "Phase 1/7"), Spacer(1, 6),
              para("Sebelum korelasi apa pun dihitung, data harus layak: kapan tiap seri mulai, apakah ada tahun yang bolong, "
                   "harga salah (bad tick), dan apakah data Open IHSG bisa dipakai. Korelasi Pearson sangat sensitif "
                   "terhadap outlier, sehingga satu harga salah bisa mengalahkan ratusan hari normal."),
              pipeline("Yahoo Finance: 12 seri OHLCV, 2007–sekarang (cache <font name='DV-M'>data/raw</font>)",
                       "Cek mulai, obs/tahun, gap, return nol, gerakan besar; sanity check 2008; 4 cek data Open",
                       "<font name='DV-M'>reports/phase1/</font>: data_quality, sanity_check_2008, decisions.json"), Spacer(1, 8)]
    oq = csv("phase1/open_quality_by_year.csv", index_col=0)
    bad_years = oq[oq["open_eq_prev_close"] > cfg["quality"]["open_checks"]["open_eq_prev_close_pct"]]["open_eq_prev_close"]
    story += [figure(1, "availability", "Ketersediaan data per seri per tahun",
                     "warna lebih gelap = lebih banyak hari berharga; sel kosong = seri belum ada (EIDO baru listing 2010); "
                     "2007 Nifty/Brent dan 2026 adalah tahun parsial."),
              figure(1, "open_quality", "Kualitas data Open ^JKSE per tahun",
                     "batang di atas garis putus-putus = tahun yang melanggar batas; yang menentukan adalah panel kedua "
                     "(Open = Close kemarin).", 70 * mm),
              figure(1, "jkse_2008", "IHSG 2007–2009 vs angka referensi 2008",
                     "titik oranye = angka referensi; harus jatuh tepat di garis IHSG Yahoo. Area abu-abu = suspend BEI.", 55 * mm)]
    dq = csv("phase1/data_quality.csv")
    rows = [["Seri", "Mulai", "Obs", "Min obs/thn", "% ret. nol", "Bad tick", "Status"]]
    for r in dq.itertuples():
        rows.append([r.name.replace("&", "&amp;"), r.first_date, f"{r.n_obs:,}".replace(",", "."), int(r.min_obs_year), f2(r.pct_zero, 2),
                     int(r.n_bad_tick), r.status])
    story += table(rows, [0.19, 0.11, 0.07, 0.08, 0.07, 0.06, 0.42], title="Hasil kunci · kualitas data per seri")
    sc = csv("phase1/sanity_check_2008.csv")
    fail = sc[sc["status"] == "FAIL"]
    ins = [f"Periode analisis mulai {d1['analysis_start']} (tanpa fallback): {d1['start_reason'].split('->')[0].strip()}.",
           f"{(dq['status'] == 'OK').sum()} dari {len(dq)} seri lolos semua cek; masalah utama di USD/IDR "
           f"({d1['bad_ticks_per_series'].get(fx, 0)} dugaan bad tick).",
           f"Sanity check 2008: {(sc['status'] == 'PASS').sum()} dari {len(sc)} PASS; FAIL {fail['tanggal'].iloc[0]} "
           f"({fail['data_yahoo'].iloc[0]} vs {fail['referensi'].iloc[0]}) karena Yahoo kehilangan hari bursa 29 Sep 2008.",
           f"Data Open gagal: {d1['open_reason'].split('(')[1].split(')')[0].strip().replace('.', ',')}; masalah terkonsentrasi di "
           f"{', '.join(str(y) for y in bad_years.index)} → <b>{d1['return_mode']}</b>."]
    rules = ["Screening seri dikunci di 2010–2019.", "Hari suspend BEI (9–10 Okt 2008) = libur, bukan data bolong.",
             "Modul gap/intraday hanya kalau data Open lolos 4 cek (README poin 7); kalau tidak, close-to-close."]
    story += code(quality.open_module_decision, "src/quality.py", 11) + [boxes(ins, rules), PageBreak()]
    checks += [("sanity FAIL Yahoo", fail["data_yahoo"].iloc[0].replace("-", "−").replace(".", ","))]

    # ---------- Phase 2 ----------
    rt = csv("phase2/usdidr_retest.csv")
    log = csv("phase2/cleaning_log.csv")
    st = csv("phase2/stale_quote_flags.csv")
    story += [SectionBand("Cleaning, alignment & returns", "Phase 2/7"), Spacer(1, 6),
              para("Data mentah diubah menjadi return yang siap dianalisis: bad tick FX dibuang dengan aturan yang dikunci "
                   "(Amandemen 1), hari suspend dan baris basi diperlakukan sebagai libur, dan setiap seri dipasangkan dengan "
                   "IHSG hanya di tanggal yang sama-sama buka (tanpa forward-fill). Di sini juga dijalankan uji ulang USD/IDR."),
              pipeline("Panel OHLCV Phase 1 + aturan cleaning (README poin 4, 8)",
                       "Buang bad tick FX; libur; return log per pasangan (harian) & W-FRI (mingguan); IHSG dalam USD",
                       "<font name='DV-M'>data/processed/</font> (parquet), cleaning_log, usdidr_retest, decisions.json"),
              Spacer(1, 8),
              figure(2, "fx_cleaning", "USD/IDR sebelum vs sesudah cleaning",
                     "titik oranye = harga yang dibuang; garis abu-abu = data mentah dengan lonjakan semu, garis biru = setelah cleaning.", 80 * mm),
              figure(2, "return_distribution", "Distribusi return harian IHSG vs normal",
                     "batang di luar kurva oranye dan titik QQ yang menjauh dari garis di kedua ujung = ekor gemuk: "
                     "hari ekstrem jauh lebih sering daripada yang diprediksi distribusi normal.", 80 * mm)]
    rows = [["Uji ulang USD/IDR (mingguan 2010–2019)", "r", "|r| 3 desimal", "Minggu", "Lolos ≥ 0,300"]]
    rows += [[r.varian, f2(r.r, 4), f2(r.abs_r_3dp, 3), r.n_minggu, "ya" if r.lolos else "tidak"] for r in rt.itertuples()]
    by = log.groupby(["id", "reason"]).size()
    rows += [["Data dibuang / ditandai", "", "", "Hari", ""]]
    rows += [[f"{i}: {reason}", "", "", n, ""] for (i, reason), n in by.items()]
    rows += [[f"{fx}: kuotasi basi (hanya ditandai)", "", "", len(st), ""]]
    story += table(rows, [0.5, 0.12, 0.13, 0.1, 0.15], title="Hasil kunci · uji ulang USD/IDR & cleaning")
    dec = d2["usdidr_retest"]
    ins = [f"Uji ulang USD/IDR: |r| = {f2(dec['abs_r'], 3)} ≥ 0,300 → <b>{dec['status']}</b>; aturan lama memberi "
           f"{f2(rt.iloc[1]['abs_r_3dp'], 3)} (tidak lolos) — status berbeda, ditulis eksplisit.",
           "Seluruh perbedaan berasal dari dua harga Jumat yang salah (29 Nov & 6 Des 2013).",
           f"{len(st)} kuotasi basi 9.612,5 hanya ditandai; tidak ada yang menjadi harga mingguan.",
           "Return harian IHSG berekor gemuk: alasan memakai HAC, Spearman, dan uji yang tidak mengandalkan normalitas."]
    rules = ["<b>Amandemen 1:</b> bad tick FX = |log return| > 5% yang dibalik ≥ 50% dalam ≤ 5 hari bursa.",
             "Batas uji ulang |r| ≥ 0,300, 3 desimal tanpa pembulatan ke atas; aturan lama = sensitivitas.",
             "Non-FX: gerakan besar hanya ditandai, tidak dibuang (README poin 8).",
             "Baris basi target (Close = kemarin & Volume 0) = libur."]
    story += code(preprocess.fx_bad_ticks, "src/preprocess.py") + [boxes(ins, rules), PageBreak()]
    checks += [("|r| USD/IDR Amandemen 1", f2(dec["abs_r"], 3)), ("|r| aturan lama", f2(rt.iloc[1]["abs_r_3dp"], 3))]

    # ---------- Phase 3 ----------
    asym = csv("phase3/asimetris_penuh.csv")
    a_key = asym[(asym["kondisi"] == asym["id"]) & (asym["mode"] == "tanda")].set_index("id")
    sub = csv("phase3/usdidr_subperiode.csv")
    s3 = js("phase3/summary.json")
    pas = kor[(kor["frekuensi"] == "mingguan") & (kor["metode"] == "pearson") & (kor["periode"] == "pasca")].set_index("id")
    story += [SectionBand("Korelasi statis, beta & korelasi asimetris", "Phase 3/7"), Spacer(1, 6),
              para("Seberapa kuat tiap seri Core bergerak bersama IHSG, apakah konsisten antar-periode dan frekuensi, dan "
                   "apakah IHSG lebih ikut jatuh daripada ikut naik? Semua hasil adalah <b>asosiasi in-sample</b>, "
                   "bukan sinyal; tabel dengan banyak uji memakai koreksi Benjamini–Hochberg."),
              pipeline("Panel return harian & mingguan (<font name='DV-M'>data/processed</font>)",
                       "Pearson/Spearman + CI Fisher; regresi faktor mingguan (HAC); korelasi kondisional vs normal + bootstrap",
                       "<font name='DV-M'>reports/phase3/</font>: korelasi, regresi_faktor_*, asimetris_*"), Spacer(1, 8),
              figure(3, "corr_heatmap", "Matriks korelasi mingguan IHSG & seri Core, 2010–2019",
                     "merah = bergerak searah, biru = berlawanan; baris/kolom IHSG menunjukkan keterkaitan langsung, "
                     "sel lain menunjukkan tumpang tindih antar-seri (sumber multikolinearitas).", 100 * mm),
              figure(3, "beta_forest", "Beta multi-faktor (CI 95%) vs beta satu faktor",
                     "titik biru + garis = beta dengan seri lain dikendalikan; berlian oranye = beta sendirian. Tanda yang "
                     "berbalik (S&amp;P 500, Hang Seng) adalah efek tumpang tindih, bukan efek negatif sungguhan.", 112 * mm),
              figure(3, "asymmetric", "Korelasi dengan IHSG saat S&amp;P 500 turun vs naik (mingguan, 2007–2026)",
                     "jarak oranye–biru = asimetri mentah; sebagian disebabkan pemotongan sampel, jadi yang ditafsirkan "
                     "adalah asimetri berlebih vs normal (tabel).", 100 * mm)]
    rows = [["Seri", "r mingguan 2010–19", "r 2020+", "Beta multi [CI 95%]", "p (BH)", "Beta 1 faktor"]]
    for i in core:
        rows.append([N[i] + (" *" if R.series_table(cfg).set_index('id').loc[i, 'label'] else ""), f2(wk.loc[i, "r"], 2, True),
                     f2(pas.loc[i, "r"], 2, True), f"{f2(reg.loc[i, 'beta'], 2, True)} [{f2(reg.loc[i, 'ci_lo'], 2, True)}; {f2(reg.loc[i, 'ci_hi'], 2, True)}]",
                     f2(reg.loc[i, "p_bh"], 3), f2(one.loc[i, "beta_1f"], 2, True)])
    story += table(rows, [0.2, 0.15, 0.12, 0.26, 0.1, 0.17], title="Hasil kunci · korelasi & beta (* = dipilih dengan pengetahuan periode uji)")
    clean = sub[sub["varian"].str.startswith("bersih")].iloc[0]
    raw = sub.iloc[0]
    ins = [f"Paling kuat (mingguan 2010–2019): {top_list}; "
           f"R² model 10 faktor {f2(s3['model_regresi']['screening']['r2'])}.",
           f"Rupiah melemah 1% ↔ IHSG {f2(reg.loc[fx, 'beta'], 2, True)}% (seri lain dikendalikan) vs "
           f"{f2(one.loc[fx, 'beta_1f'], 2, True)}% sendirian.",
           f"Asimetri berlebih IHSG–EEM {f2(a_key.loc['EEM', 'asimetri_berlebih'], 2, True)} (CI bootstrap "
           f"{f2(a_key.loc['EEM', 'ci_lo_boot'], 2, True)} s/d {f2(a_key.loc['EEM', 'ci_hi_boot'], 2, True)}): IHSG lebih ikut jatuh.",
           f"Hipotesis \"korelasi rupiah menguat\" dicabut: mentah {f2(raw.r_awal, 2, True)} → {f2(raw.r_akhir, 2, True)} "
           f"(p &lt; 0,001), bersih {f2(clean.r_awal, 2, True)} → {f2(clean.r_akhir, 2, True)} (p = {f2(clean.p_fisher, 2)})."]
    rules = ["Fase deskriptif: tidak ada pemilihan ulang seri.", "Benjamini–Hochberg untuk tabel banyak uji; "
             "istilah \"asosiasi\" untuk hasil in-sample.", "HAC lag = ⌊4(T/100)^(2/9)⌋; VIF > 5 → tampilkan beta satu faktor.",
             "Brent selalu berlabel \"dipilih dengan pengetahuan periode uji\"."]
    story += code(analysis.expected_conditional_corr, "src/analysis.py") + [boxes(ins, rules), PageBreak()]
    checks += [("R² multi", f2(s3["model_regresi"]["screening"]["r2"])),
               ("asimetri EEM", f2(a_key.loc["EEM", "asimetri_berlebih"], 2, True)),
               ("hipotesis bersih awal", f2(clean.r_awal, 2)), ("hipotesis p bersih", f2(clean.p_fisher, 2))]

    # ---------- Phase 4 ----------
    pk = csv("phase4/ccf_puncak.csv", index_col=0)
    l1 = csv("phase4/lead_satu_faktor.csv", index_col=0)
    gr = csv("phase4/granger.csv", index_col=0)
    em4 = csv("phase4/eido_model.csv", index_col=0)
    s4 = js("phase4/summary.json")
    co = s4["eido_kointegrasi"]
    story += [SectionBand("Lead-lag, presedensi prediktif & sinyal semalam EIDO", "Phase 4/7"), Spacer(1, 6),
              para("Apakah yang terjadi di New York malam ini mendahului IHSG besok? Diuji dengan cross-correlation per lag, "
                   "regresi lead (HAC), Granger sebagai <b>presedensi prediktif (bukan sebab-akibat)</b>, dan premium EIDO "
                   "terhadap IHSG dalam USD. Semua estimasi hanya memakai data s/d 31 Des 2019."),
              pipeline("Return harian pasangan 2010–2019 (kuotasi basi USD/IDR dibuang)",
                       "CCF k = −5..+5; regresi lead 1 & multi faktor; Granger (BIC); kointegrasi & premium EIDO",
                       "<font name='DV-M'>reports/phase4/</font>: ccf, granger, lead_*, eido_*"), Spacer(1, 8),
              figure(4, "ccf", "Korelasi IHSG(t+k) vs seri(t), harian 2010–2019",
                     "kolom +0 = hari yang sama, +1 = seri mendahului IHSG satu hari; seri sinkron menyala di +0, "
                     "EEM/S&amp;P 500/VIX di +1, rupiah di −1 (IHSG mendahului rupiah).", 95 * mm),
              figure(4, "eido_premium", "Premium log EIDO − log IHSG dalam USD, 2010–2019",
                     "level bergeser turun perlahan (EIDO memuat dividen, IHSG indeks harga) dengan simpangan jangka pendek; "
                     "bagian harian yang belum masuk close IHSG (Δpremium) yang dipakai sebagai fitur.", 60 * mm)]
    rows = [["Seri", "Sesi", "k puncak", "r (k=0)", "r (k=+1)", "Beta lead 1 faktor", "Granger seri→IHSG p (BH)"]]
    for i in core:
        rows.append([N[i], pk.loc[i, "session"], int(pk.loc[i, "k_puncak"]), f2(pk.loc[i, "r_k0"], 3, True), f2(pk.loc[i, "r_k1"], 3, True),
                     f2(l1.loc[i, "beta"], 3, True) if i in l1.index else "– (sinkron)", p_txt(gr.loc[i, "p_seri_ke_target_bh"])])
    story += table(rows, [0.2, 0.08, 0.1, 0.12, 0.12, 0.18, 0.2], title="Hasil kunci · lead-lag & Granger (in-sample 2010–2019)")
    plus = em4.loc["pembanding + Δpremium"]
    ins = [f"Seri sinkron puncak di k = 0; EEM ({f2(pk.loc['EEM', 'r_k1'], 2, True)}), S&amp;P 500 dan VIX puncak di k = +1.",
           f"Rupiah puncak di k = −1 ({f2(pk.loc[fx, 'r_puncak'], 2, True)}): IHSG mendahului rupiah, bukan sebaliknya.",
           f"Δpremium EIDO menambah R² {f2(plus.tambahan_r2_vs_base, 3, True)} di atas S&amp;P 500 + EEM (Wald p {p_txt(plus.p_tambahan)}).",
           f"Kointegrasi EIDO: Engle–Granger konstanta p = {f2(co['eg_c_p'], 2)} (tidak terkointegrasi menurut aturan) → "
           "fitur yang dipakai Δpremium."]
    rules = ["Kebersihan OOS: semua estimasi s/d 31 Des 2019; data 2020+ tidak disentuh.",
             "Harga USD/IDR ber-flag_stale_quote dibuang di semua analisis harian.",
             "Granger = presedensi prediktif, bukan sebab-akibat; lag via BIC (AIC pembanding).",
             "Fitur EIDO utama = Δpremium; z-spread versi tren hanya sensitivitas berlabel."]
    story += code(leadlag.cross_correlation, "src/leadlag.py") + [boxes(ins, rules), PageBreak()]
    checks += [("EEM k+1", f2(pk.loc["EEM", "r_k1"], 2)), ("tambahan R² Δpremium", f2(plus.tambahan_r2_vs_base, 3, True)),
               ("EG p", f2(co["eg_c_p"], 2))]

    # ---------- Phase 5 ----------
    rr = csv("phase5/rolling_ringkas_52.csv", index_col=0)
    q10 = csv("phase5/brent_kuantil_2010_2019.csv")
    qall = csv("phase5/brent_kuantil_penuh.csv")
    sp = csv("phase5/brent_lonjakan_2010_2019.csv", index_col=0)
    story += [SectionBand("Regime & krisis", "Phase 5/7"), Spacer(1, 6),
              para("Apakah keterkaitan IHSG berubah dari waktu ke waktu, dan bagaimana IHSG berperilaku saat guncangan global? "
                   "Fase ini <b>deskriptif</b> (seluruh periode boleh dipakai) dan tidak boleh mengubah spesifikasi Phase 6."),
              pipeline("Return mingguan & harga harian 2007–2026; emas, DXY, ^TNX (analisis saja)",
                       "Rolling 52/156 minggu; event study [−20, +60] hari bursa, 6 jangkar; quantile regression Brent",
                       "<font name='DV-M'>reports/phase5/</font>: rolling_*, event_study_*, brent_*, pelindung_*"), Spacer(1, 8),
              figure(5, "rolling", "Korelasi bergulir IHSG (52 & 156 minggu) dengan episode krisis diarsir",
                     "garis biru (52 minggu) bereaksi cepat, oranye (156 minggu) menunjukkan tren; penurunan yang bertahan "
                     "di keduanya = tren, lonjakan yang kembali = sesaat.", 95 * mm),
              figure(5, "event_paths", "Event study: harga di sekitar jangkar krisis (close sebelum hari 0 = 100)",
                     "hari 0 = jangkar; garis tebal = IHSG. Panel 2026 hanya deskriptif dan terpotong akhir data.", 100 * mm),
              figure(5, "quantile", "Koefisien Brent per kuantil return IHSG (CI 95%)",
                     "kalau minyak menekan IHSG di minggu buruk, titik di kuantil 0,1 akan negatif; yang terlihat: nol (2010–2019) "
                     "atau sedikit positif (seluruh periode).", 70 * mm),
              figure(5, "havens", "Aset pelindung: median perubahan sampai titik terendah IHSG",
                     "batang positif = naik ketika IHSG mencapai titik terendah; batang oranye = IHSG sebagai pembanding.", 62 * mm)]
    rows = [["Event study (median 6 episode)", "Drawdown maks", "Hari ke titik terendah", "Return +60 hari", "Korelasi: setahun sebelum → krisis"]]
    for i in ["JKSE", "STI", "KLCI", "EEM"]:
        r = em.loc[i]
        rows.append([N[i], f2(r.max_drawdown, 1, pct=True), f2(r.hari_ke_titik_terendah, 1) if r.hari_ke_titik_terendah % 1 else int(r.hari_ke_titik_terendah),
                     f2(r.return_kumulatif_akhir, 1, pct=True),
                     "–" if pd.isna(r.r_setahun_sebelum) else f"{f2(r.r_setahun_sebelum, 2, True)} → {f2(r.r_jendela, 2, True)}"])
    rows += [["Rolling 52 minggu: rata-rata pra-2020 / 2020–22 / 2023+", "", "", "", ""]]
    for i in ["HSI", "SPX"]:
        rows.append([N[i], "", "", "", f"{f2(rr.loc[i, 'rata_pra'], 2)} / {f2(rr.loc[i, 'rata_pasca_awal'], 2)} / {f2(rr.loc[i, 'rata_pasca_akhir'], 2)}"])
    bq = lambda t, q: t[(t["variabel"] == "BRENT") & (t["kuantil"].astype(str) == q)].iloc[0]  # noqa: E731
    rows += [["Brent τ = 0,1 (2010–19 / seluruh periode)", "", "", "", f"{f2(bq(q10, '0.1').beta, 3, True)} / {f2(bq(qall, '0.1').beta, 3, True)}"]]
    story += table(rows, [0.32, 0.14, 0.16, 0.14, 0.24], title="Hasil kunci · krisis, rolling, Brent")
    total = sp.loc[[i for i in sp.index if "total" in i][0]]
    ins = [f"Hang Seng: {f2(rr.loc['HSI', 'rata_pra'], 2)} → {f2(rr.loc['HSI', 'rata_pasca_akhir'], 2)} setelah 2023 (tren); "
           f"S&amp;P 500: lonjakan {f2(rr.loc['SPX', 'rata_pasca_awal'], 2)} di 2020–22 lalu kembali (sesaat).",
           f"IHSG jatuh {f2(em.loc['JKSE', 'max_drawdown'], 1, pct=True)} (median) — lebih dalam dari STI dan KLCI; "
           "korelasi regional naik saat krisis.",
           f"Brent tidak menekan IHSG di ekor bawah; minggu lonjakan: kemiringan total {f2(total.beta, 2, True)} "
           f"(CI {f2(total.ci_lo, 2, True)} s/d {f2(total.ci_hi, 2, True)}), tanda tidak berbalik.",
           f"Pelindung investor rupiah: USD {f2(hv[fx], 1, True, pct=True)}, emas dalam rupiah {f2(hv['GOLD_IDR'], 1, True, pct=True)}; "
           f"emas dalam USD {f2(hv['GOLD'], 1, True, pct=True)}."]
    rules = ["Phase 5 deskriptif; tidak boleh mengubah Spesifikasi Phase 6.",
             "Jangkar = guncangan global yang menular ke EM; jendela [−20, +60]; 2018 = 10 Agu (5 Feb = sensitivitas).",
             "Okt 2026 hanya deskriptif, dikeluarkan dari median.", "Hasil Brent berlabel \"dipilih dengan pengetahuan periode uji\"."]
    story += code(regime.episode_metrics, "src/regime.py") + [boxes(ins, rules), PageBreak()]
    checks += [("HSI pasca akhir", f2(rr.loc["HSI", "rata_pasca_akhir"], 2)), ("drawdown KLCI", f2(em.loc["KLCI", "max_drawdown"], 1, pct=True))]

    # ---------- Phase 6 ----------
    plac = csv("phase6/placebo_target_acak.csv")
    leak = csv("phase6/placebo_bocor.csv", index_col=0)
    h6 = csv("phase6/vol_har_vs_harx.csv", index_col=0)
    oc = csv("phase6/open_to_close_metrik.csv", index_col=0)
    val = csv("phase6/validasi_garch_vs_arch.csv", index_col=0)
    ps = js("phase6/summary.json")["placebo_ringkas"]          # rata-rata dari nilai tanpa pembulatan
    story += [SectionBand("Uji out-of-sample: arah & volatilitas", "Phase 6/7", ORANGE), Spacer(1, 6),
              para("Satu-satunya phase yang boleh menghasilkan klaim \"sinyal\". Model dilatih ulang tiap bulan hanya dengan "
                   "data yang sudah tersedia saat itu (walk-forward, expanding sejak 2010) dan diuji pada 2020–2026, mengikuti "
                   "spesifikasi yang dikunci sebelum Phase 5. Uji placebo membuktikan tidak ada kebocoran."),
              pipeline("Fitur hari t (EEM, S&amp;P 500, ΔVIX, Δpremium EIDO, IHSG t); return IHSG harian",
                       "Logit vs naif, refit bulanan; DM (log loss); GARCH-X & HAR-X vs tanpa VIX; placebo",
                       "<font name='DV-M'>reports/phase6/</font>: arah_*, vol_*, placebo_*, validasi_garch_vs_arch"), Spacer(1, 8),
              figure(6, "walkforward", "Skema walk-forward (contoh satu refit per tahun)",
                     "batang biru = data latih yang terus memanjang; potongan oranye = bulan yang diprediksi dengan parameter "
                     "dari data sebelum bulan itu.", 70 * mm),
              figure(6, "calibration", "Kalibrasi probabilitas arah (OOS)",
                     "titik di garis diagonal = probabilitas jujur; titik paling kanan di bawah diagonal = model terlalu yakin "
                     "di hari ekstrem, penyebab log loss tidak lebih baik dari naif.", 80 * mm),
              figure(6, "vol_forecast", "Prakiraan volatilitas GARCH-X vs GARCH vs |return| realisasi (OOS)",
                     "garis oranye (dengan VIX) bereaksi lebih cepat saat VIX melonjak; keunggulannya diuji dengan QLIKE (tabel).", 70 * mm),
              figure(6, "placebo", "Uji placebo: target diacak (10x) vs model asli",
                     "titik biru tersebar di sekitar nol/51–55% (tanpa kebocoran); garis oranye = model asli. Selisih log loss model "
                     "asli masih di dalam sebaran placebo, hit rate-nya di luar.", 50 * mm)]
    rows = [["Uji (OOS 2020–2026)", "Naif / tanpa VIX", "Model", "p (DM) / catatan"],
            ["Arah: hit rate", f2(hr(m6.loc["baseline", "hit_rate"], m6.loc["baseline", "n"]), 1, pct=True), f2(hr(m6.loc["utama", "hit_rate"], m6.loc["utama", "n"]), 1, pct=True), "pelengkap"],
            ["Arah: AUC", f2(m6.loc["baseline", "auc"], 3), f2(m6.loc["utama", "auc"], 3), "pelengkap"],
            ["Arah: Brier", f2(m6.loc["baseline", "brier"], 4), f2(m6.loc["utama", "brier"], 4), "pelengkap"],
            ["<b>Arah: log loss (uji yang dikunci)</b>", f2(m6.loc["baseline", "log_loss"], 4), f2(m6.loc["utama", "log_loss"], 4),
             f"<b>p = {f2(dm6.loc['utama', 'p'], 3)} → tidak terbukti</b>"],
            ["<b>GARCH-X vs GARCH: QLIKE</b>", f2(g6.loc["QLIKE", "tanpa_vix"], 4), f2(g6.loc["QLIKE", "dengan_vix"], 4),
             f"<b>p = {f2(g6.loc['QLIKE', 'p'], 3)} → VIX membantu</b>"],
            ["HAR-X vs HAR: QLIKE (mingguan)", f2(h6.loc["QLIKE", "tanpa_vix"], 4), f2(h6.loc["QLIKE", "dengan_vix"], 4), f"p = {f2(h6.loc['QLIKE', 'p'], 3)}"],
            ["Placebo target acak (rata-rata 10x)", "", f"selisih log loss {f2(ps['rata_selisih_log_loss'], 4, True)}", f"AUC rata-rata {f2(ps['rata_auc'], 3)}"],
            ["Placebo bocor: semua fitur t+1", "", f2(hr(leak.loc["bocor: semua fitur t+1", "hit_rate"], leak.loc["bocor: semua fitur t+1", "n"]), 1, pct=True), "melonjak (terdeteksi)"],
            ["Placebo bocor halus: STI/KLCI/HSI t+1", "", f2(hr(leak.iloc[2]["hit_rate"], leak.iloc[2]["n"]), 1, pct=True), "melonjak (terdeteksi)"],
            ["Open-to-close (EKSPLORATIF): hit rate", f2(hr(oc.loc["baseline", "hit_rate"], oc.loc["baseline", "n"]), 1, pct=True), f2(hr(oc.loc["utama_oc", "hit_rate"], oc.loc["utama_oc", "n"]), 1, pct=True), "tidak signifikan"],
            ["Validasi GARCH sendiri vs arch (γ = 0)", "", "", f"loglik identik (selisih {f2(val.iloc[-1]['selisih_abs'], 1)})"]]
    story += table(rows, [0.38, 0.16, 0.2, 0.26], header_color=ORANGE, title="Hasil kunci · OOS")
    ins = ["<b>Arah IHSG besok: tidak terbukti</b> lebih baik dari tebakan naif menurut uji yang dikunci, walau hit rate lebih tinggi.",
           "Penyebab: terlalu yakin di hari ekstrem (kalibrasi) — log loss 2020–2022 bahkan lebih buruk dari naif.",
           "<b>Volatilitas: VIX terbukti membantu</b> (GARCH-X dan HAR-X lolos DM QLIKE).",
           "Pipeline bersih: placebo acak ≈ naif; kebocoran yang disengaja langsung terdeteksi."]
    rules = ["Spesifikasi Phase 6 dikunci sebelum Phase 5: fitur, target, OOS 2020+, refit bulanan, DM log loss.",
             "Logit tanpa regularisasi (tanpa tuning); GARCH-X perkalian exp(γ·ΔlogVIX), divalidasi vs arch.",
             "Uji tambahan (Brier, uji tanda) hanya penjelas, tidak mengubah vonis.",
             "Tradeability: daya prediksi close-to-close TIDAK berarti dapat dieksploitasi."]
    story += code(oos.walk_forward, "src/oos.py") + [boxes(ins, rules), PageBreak()]
    checks += [("placebo rata-rata selisih", f2(ps["rata_selisih_log_loss"], 4)), ("HAR p", f2(h6.loc["QLIKE", "p"], 3)), ("open-to-close", f2(hr(oc.loc["utama_oc", "hit_rate"], oc.loc["utama_oc", "n"]), 1, pct=True)),
               ("placebo bocor penuh", f2(hr(leak.loc["bocor: semua fitur t+1", "hit_rate"], leak.loc["bocor: semua fitur t+1", "n"]), 1, pct=True)),
               ("placebo bocor halus", f2(hr(leak.iloc[2]["hit_rate"], leak.iloc[2]["n"]), 1, pct=True))]

    # ---------- Phase 7 ----------
    bt = csv("phase7/bigcap_perbandingan.csv", index_col=0)
    ch = csv("phase7/bigcap_juara.csv", index_col=0)
    story += [SectionBand("Demonstrasi pipeline big cap & ringkasan", "Phase 7/7", AQUA), Spacer(1, 6),
              para("Pipeline dijalankan ulang untuk BBCA, BBRI, TLKM, PTBA, dan ASII lewat <font name='DV-M'>project.target_id</font> "
                   "— Core, aturan, periode, dan aturan OOS sama dengan IHSG, tanpa seleksi ulang. Aturan dividen besar "
                   "(> 10% harga) berlaku untuk saham. Hasil deskriptif, bukan rekomendasi saham."),
              pipeline("Harga 5 saham + dividen; panel Core; IHSG bersih sebagai faktor pasar",
                       "Phase 1–2 per target; beta HAC; asimetri vs EEM; lead; event study; GARCH-X (IS & OOS)",
                       "<font name='DV-M'>reports/phase7/</font> & <font name='DV-M'>reports/targets/&lt;id&gt;/</font>"), Spacer(1, 8),
              figure(7, "bigcap_betas", "Beta mingguan satu faktor: 2010–2019 vs 2020+",
                     "biru = 2010–2019, oranye = 2020+; jarak antar-titik = perubahan sensitivitas (mis. rupiah TLKM, EEM PTBA).", 75 * mm),
              figure(7, "bigcap_drawdown", "Median drawdown big cap di 6 episode krisis",
                     "batang lebih panjang = jatuh lebih dalam; garis oranye = median IHSG sebagai pembanding.", 62 * mm)]
    rows = [["Saham", "β IHSG 10–19 / 20+", "β USD/IDR 10–19 / 20+", "β EEM 10–19 / 20+", "Asimetri vs EEM [CI]", "Draw-down krisis", "γ VIX (IS) / DM p (OOS)"]]
    for i, r in bt.iterrows():
        rows.append([N[i], f"{f2(r['beta_ihsg_2010_2019'], 2)} / {f2(r['beta_ihsg_2020+'], 2)}",
                     f"{f2(r['beta_usdidr_2010_2019'], 2)} / {f2(r['beta_usdidr_2020+'], 2)}",
                     f"{f2(r['beta_eem_2010_2019'], 2)} / {f2(r['beta_eem_2020+'], 2)}",
                     f"{f2(r['asimetri_berlebih'], 2, True)} [{f2(r['asimetri_ci_lo'], 2, True)}; {f2(r['asimetri_ci_hi'], 2, True)}]",
                     f2(r["krisis_drawdown_median"], 1, pct=True), f"{f2(r['garch_gamma_is'], 2)} / {f2(r['garch_dm_qlike_p'], 2)}"])
    story += table(rows, [0.16, 0.12, 0.15, 0.12, 0.18, 0.12, 0.15], header_color=AQUA, title="Hasil kunci · perbandingan big cap")
    ins = [f"{d.split('(')[0].strip()[0].upper()}{d.split('(')[0].strip()[1:]}: <b>{N[r.saham]}</b> ({f2(float(r.nilai), 3, True)})."
           for d, r in ch.iterrows()][:4]
    ins.append(f"VIX: γ in-sample signifikan di {(bt['garch_lr_p_is'] < 0.05).sum()} dari {len(bt)} saham, tetapi "
               f"{(bt['garch_dm_qlike_p'] < 0.05).sum()} lolos uji OOS per saham. Peringkat deskriptif; banyak CI tumpang tindih.")
    rules = ["Tidak ada seleksi ulang seri Core; spesifikasi & aturan OOS sama dengan IHSG.",
             "Aturan dividen besar: return ex-date dengan dividen > 10% close sebelumnya = NaN (README poin 9).",
             "Target diganti lewat config (project.target_id); output di reports/targets/&lt;id&gt;/.",
             "ADRO keluar (patahan Nov 2024), ANTM keluar (data kotor); PTBA pengganti."]
    story += code(preprocess.large_dividend_days, "src/preprocess.py") + [boxes(ins, rules), PageBreak()]
    checks += [("BBRI beta rupiah", f2(bt.loc["BBRI", "beta_usdidr_2010_2019"], 2)), ("TLKM asimetri", f2(bt.loc["TLKM", "asimetri_berlebih"], 2, True))]

    # ---------- Penutup ----------
    flog = ROOT / "reports" / "forward_test" / "log.csv"
    n_log = len(pd.read_csv(flog)) if flog.exists() else 0
    comp = json.loads((ROOT / "models" / "frozen_2026-10-21_pembanding.json").read_text())
    story += [SectionBand("Vonis pra-registrasi, keterbatasan & forward test", "Penutup"), Spacer(1, 6),
              para("Vonis pra-registrasi (uji yang dikunci)", "h2"),
              *table([["Pertanyaan", "Uji yang dikunci", "Hasil", "Vonis"],
                      ["Arah IHSG besok dari data semalam", "DM log loss vs naif, OOS 2020–2026",
                       f"{f2(m6.loc['utama', 'log_loss'], 4)} vs {f2(m6.loc['baseline', 'log_loss'], 4)}; p = {f2(dm6.loc['utama', 'p'], 2)}",
                       "<b>Tidak terbukti</b>"],
                      ["VIX untuk volatilitas IHSG", "DM QLIKE, GARCH-X vs GARCH",
                       f"{f2(g6.loc['QLIKE', 'dengan_vix'], 3)} vs {f2(g6.loc['QLIKE', 'tanpa_vix'], 3)}; p = {f2(g6.loc['QLIKE', 'p'], 3)}",
                       "<b>Terbukti</b>"],
                      ["Uji ulang USD/IDR sebagai driver", "|r| mingguan ≥ 0,300 (Amandemen 1)", f"|r| = {f2(dec['abs_r'], 3)}", "<b>Lolos</b>"]],
                     [0.3, 0.27, 0.27, 0.16]),
              para("Keterbatasan", "h2")]
    lim = ["Yahoo Finance bukan sumber resmi: harga salah (USD/IDR 2010–2013), hari hilang (29 Sep 2008), baris basi saham, "
           "data Open tidak layak di beberapa tahun.",
           "Hanya close-to-close: tanpa Open/intraday yang layak, tradeability tidak bisa diuji dengan baik.",
           "IDR=X tutup di jam AS; EIDO melacak MSCI Indonesia (basis risk) dan memuat dividen.",
           "Sebagian besar temuan adalah asosiasi in-sample dengan banyak uji; yang \"terbukti\" hanya hasil OOS yang dikunci.",
           "Krisis hanya 6 episode; jangkar mengandung penilaian; Okt 2026 terpotong.",
           "Big cap: 5 saham, return disesuaikan dividen Yahoo; total return sendiri belum diterapkan."]
    story += [Paragraph(x, S["box"], bulletText="•") for x in lim]
    story += [para("Status forward test", "h2"),
              para(f"Model beku <font name='DV-M'>models/{comp['model_version']}.json</font> (data s/d {comp['as_of']}) dan pembanding beku "
                   f"(baseline naif {f2(comp['baseline_p_naik'], 3)}, GARCH tanpa VIX). Jadwal launchd Senin–Jumat 09:15 Melbourne "
                   f"(05:15/06:15 WIB). Baris log saat dokumen ini dibangun: <b>{n_log}</b>. Evaluasi pra-registrasi baru dilakukan "
                   "setelah 250 baris valid (DM log loss vs naif beku; DM QLIKE vs GARCH tanpa VIX), dilaporkan apa pun hasilnya.")]
    checks += [("baseline beku", f2(comp["baseline_p_naik"], 3))]

    doc = Doc(path)
    doc.multiBuild(story)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    return path, R.crosscheck(checks, readme)


def render(pdf: Path, out_dir: Path, scale: float = 1.4) -> list[Path]:
    import pypdfium2 as pdfium
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(str(pdf))
    paths = []
    for i in range(len(doc)):
        p = out_dir / f"page_{i + 1:02d}.png"
        doc[i].render(scale=scale).to_pil().save(p)
        paths.append(p)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update-notebooks", action="store_true")
    ap.add_argument("--render", type=Path, help="render tiap halaman PDF ke PNG di folder ini")
    args = ap.parse_args(argv)
    if args.update_notebooks:
        update_notebooks()
        return 0
    path, chk = build()
    import pypdfium2 as pdfium
    print(f"PDF: {path.relative_to(ROOT)} ({len(pdfium.PdfDocument(str(path)))} halaman)")
    print("Cek silang angka vs README:")
    print(chk.to_string(index=False))
    if args.render:
        print(f"Render: {len(render(path, args.render))} halaman -> {args.render}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
