"""
Relatório — melhor representação (k, N) de cada modelo na grade do TCC II × original (k=3, N=1000).

Lê todos os results_{modelo}_n{dim}.json de reports/v2/ e dos subdiretórios de representação
(reports/v2/k2/, reports/v2/k3_1101/, …) e gera reports/v2/melhor_k_n_por_modelo.pdf com uma
única tabela: para cada modelo, a configuração original, a melhor por ROC-AUC e a melhor por MCC,
com Δ em relação ao original (em verde quando maior que o desvio padrão da vencedora).

Requer reportlab (requirements.txt) e as fontes Arial do macOS
(/System/Library/Fonts/Supplemental).

Uso (a partir de biomarker_pipeline/):
    python src/relatorios/melhor_k_n_por_modelo.py
"""

import glob
import json
import os
import re
import sys
import logging

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

REPORTS_DIR = "reports/v2"
OUT = os.path.join(REPORTS_DIR, "melhor_k_n_por_modelo.pdf")
ORIGINAL = ("k3", 1000)

FONT_DIR = "/System/Library/Fonts/Supplemental"
pdfmetrics.registerFont(TTFont("Arial", f"{FONT_DIR}/Arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", f"{FONT_DIR}/Arial Bold.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", f"{FONT_DIR}/Arial Italic.ttf"))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic",
                              boldItalic="Arial-Bold")

INK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#52606D")
HEAD = colors.HexColor("#243B53")
RULE = colors.HexColor("#BCCCDC")
BEST = colors.HexColor("#E3F2EC")
ORIG = colors.HexColor("#FFF4E0")
OK = colors.HexColor("#1E7B4F")

st = {
    "cell": ParagraphStyle("cell", fontName="Arial", fontSize=8.2, leading=10.2, textColor=INK, alignment=TA_LEFT),
    "cellc": ParagraphStyle("cellc", fontName="Arial", fontSize=8.2, leading=10.2, textColor=INK, alignment=TA_CENTER),
    "cellb": ParagraphStyle("cellb", fontName="Arial-Bold", fontSize=8.2, leading=10.2, textColor=INK),
    "small": ParagraphStyle("small", fontName="Arial", fontSize=7.4, leading=9.2, textColor=MUTED),
    "headc": ParagraphStyle("headc", fontName="Arial-Bold", fontSize=8.2, leading=10.2, textColor=colors.white),
    "headcc": ParagraphStyle("headcc", fontName="Arial-Bold", fontSize=8.2, leading=10.2, textColor=colors.white,
                             alignment=TA_CENTER),
}

MODELS = [("logistic_regression", "Regressão Logística"), ("svm", "SVM"), ("random_forest", "Random Forest"),
          ("gradient_boosting", "Gradient Boosting"), ("naive_bayes", "Naive Bayes"), ("knn", "kNN")]


def c(text, style="cell"):
    return Paragraph(str(text), st[style])


def pct(x, d=1):
    return f"{x * 100:.{d}f}".replace(".", ",")


def pm(m, d=1):
    return f"{pct(m['mean'], d)} ± {pct(m['std'], d)}"


def load_all():
    """Uma entrada por results_{modelo}_n{dim}.json: representação, dimensão, métricas e hiperparâmetros."""
    rows = []
    for d in [REPORTS_DIR] + sorted(glob.glob(os.path.join(REPORTS_DIR, "k*"))):
        rep = "k3" if d == REPORTS_DIR else os.path.basename(d)
        for f in glob.glob(os.path.join(d, "results_*_n*.json")):
            m = re.search(r"results_(.+)_n(\d+)\.json$", f)
            r = json.load(open(f))
            rows.append({"rep": rep, "dim": int(m.group(2)), "proj": r.get("projection", True),
                         "model": m.group(1), "metrics": r["metrics"], "params": r["best_params"]})
    return rows


def rep_label(rep):
    k, _, pat = rep.partition("_")
    if not pat:
        return f"k={k[1:]} contínuo"
    return f"k={k[1:]} espaçado {pat.replace('+', ' + ')}"


def n_label(x):
    return f"{x['dim']:,}".replace(",", ".") if x["proj"] else f"sem projeção ({x['dim']:,})".replace(",", ".")


def params_label(model, p):
    if model == "logistic_regression":
        return f"C = {str(p['model__C']).replace('.', ',')}"
    if model == "svm":
        return f"C = {p['model__C']}, gamma = {p['model__gamma']}"
    if model == "random_forest":
        depth = "livre" if p["max_depth"] is None else p["max_depth"]
        return f"{p['n_estimators']} árvores, profundidade {depth}"
    if model == "gradient_boosting":
        return f"lr = {str(p['learning_rate']).replace('.', ',')}, {p['n_estimators']} estágios"
    if model == "naive_bayes":
        return f"var_smoothing = {p['var_smoothing']:.0e}"
    if model == "knn":
        return f"k = {p['model__n_neighbors']} vizinhos"
    return str(p)


def mean(x, k):
    return x["metrics"][k]["mean"]


def sd(x, k):
    return x["metrics"][k]["std"]


def summarize(rows):
    summary = {}
    for m, _ in MODELS:
        rs = [x for x in rows if x["model"] == m]
        summary[m] = {
            "n": len(rs),
            "orig": next(x for x in rs if (x["rep"], x["dim"]) == ORIGINAL),
            "roc": max(rs, key=lambda x: (mean(x, "roc_auc"), -sd(x, "roc_auc"))),
            "mcc": max(rs, key=lambda x: (mean(x, "mcc"), -sd(x, "mcc"))),
        }
    return summary


def delta(best, orig, k):
    d = mean(best, k) - mean(orig, k)
    txt = f"{'+' if d >= 0 else '−'}{pct(abs(d))}"
    return f"<font color='{OK.hexval()}'><b>{txt}</b></font>" if d > sd(best, k) else txt


def table_best(summary):
    head = ["Modelo", "", "Representação", "N", "ROC-AUC", "MCC", "Sens.", "Espec.", "Hiperparâmetros"]
    rows = [[c(h, "headc" if i < 4 or i == 8 else "headcc") for i, h in enumerate(head)]]
    extra = []
    for m, label in MODELS:
        s = summary[m]
        entries = [("Original", s["orig"])]
        if s["roc"] is s["mcc"]:
            entries.append(("Melhor<br/>(ROC-AUC e MCC)", s["roc"]))
        else:
            entries.append(("Melhor<br/>por ROC-AUC", s["roc"]))
            entries.append(("Melhor<br/>por MCC", s["mcc"]))
        first = len(rows)
        for j, (tag, x) in enumerate(entries):
            is_orig = tag == "Original"
            roc = pm(x["metrics"]["roc_auc"])
            mcc = pm(x["metrics"]["mcc"])
            if not is_orig:
                roc += f"<br/><font size='7'>Δ {delta(x, s['orig'], 'roc_auc')}</font>"
                mcc += f"<br/><font size='7'>Δ {delta(x, s['orig'], 'mcc')}</font>"
            rows.append([
                c(f"{label}<br/><font size='7' color='{MUTED.hexval()}'>{s['n']} configurações</font>"
                  if j == 0 else "", "cellb"),
                c(tag, "cell" if is_orig else "cellb"),
                c(rep_label(x["rep"])),
                c(n_label(x)),
                c(roc, "cellc"),
                c(mcc, "cellc"),
                c(pct(mean(x, "sensitivity")), "cellc"),
                c(pct(mean(x, "specificity")), "cellc"),
                c(params_label(m, x["params"]), "small"),
            ])
            extra.append(("BACKGROUND", (1, len(rows) - 1), (-1, len(rows) - 1), ORIG if is_orig else BEST))
        extra.append(("SPAN", (0, first), (0, len(rows) - 1)))
        extra.append(("LINEBELOW", (0, len(rows) - 1), (-1, len(rows) - 1), 0.8, HEAD))
    widths = [2.25, 1.75, 2.6, 1.75, 1.95, 1.95, 1.2, 1.25, 2.6]
    t = Table(rows, colWidths=[w * cm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, HEAD),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ] + extra))
    return t


def main():
    summary = summarize(load_all())
    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, title="Melhor k e N por modelo — TCC II",
                            author="Grupo do TCC", subject="Grade de representação k-mer do TCC II")
    doc.build([table_best(summary)])
    log.info("PDF: %s", OUT)
    for m, _ in MODELS:
        s = summary[m]
        log.info("%-20s original MCC %.1f | melhor ROC-AUC: %s n%d | melhor MCC: %s n%d", m,
                 mean(s["orig"], "mcc") * 100, s["roc"]["rep"], s["roc"]["dim"], s["mcc"]["rep"], s["mcc"]["dim"])


if __name__ == "__main__":
    main()
