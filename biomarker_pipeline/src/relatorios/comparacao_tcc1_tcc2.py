"""
Relatório — comparação TCC I × TCC II (dados e resultados da linha de base, N = 1000).

Gera reports/v2/comparacao_tcc1_tcc2.pdf a partir dos arquivos versionados:
contagens de genes e sequências por grupo, métricas da validação cruzada (N = 1000),
ROC-AUC por N e predições das 5 proteínas de validação externa, nos dois datasets.

Requer reportlab (requirements.txt) e as fontes Arial do macOS
(/System/Library/Fonts/Supplemental).

Uso (a partir de biomarker_pipeline/):
    python src/relatorios/comparacao_tcc1_tcc2.py
"""

import csv
import glob
import json
import os
import sys
import logging
from collections import defaultdict

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

P = "."
OUT = os.path.join(P, "reports/v2/comparacao_tcc1_tcc2.pdf")

FONT_DIR = "/System/Library/Fonts/Supplemental"
pdfmetrics.registerFont(TTFont("Arial", f"{FONT_DIR}/Arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", f"{FONT_DIR}/Arial Bold.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", f"{FONT_DIR}/Arial Italic.ttf"))
pdfmetrics.registerFont(TTFont("ArialUni", f"{FONT_DIR}/Arial Unicode.ttf"))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic",
                              boldItalic="Arial-Bold")

INK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#52606D")
HEAD = colors.HexColor("#243B53")
RULE = colors.HexColor("#BCCCDC")
ZEBRA = colors.HexColor("#F0F4F8")
TCC2 = colors.HexColor("#E3F2EC")
OK = colors.HexColor("#1E7B4F")
BAD = colors.HexColor("#B42318")
WARN = colors.HexColor("#9A6700")

st = {
    "title": ParagraphStyle("title", fontName="Arial-Bold", fontSize=18, leading=22, textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("sub", fontName="Arial", fontSize=9.5, leading=13, textColor=MUTED, spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName="Arial-Bold", fontSize=12.5, leading=16, textColor=HEAD, spaceBefore=12, spaceAfter=5),
    "body": ParagraphStyle("body", fontName="Arial", fontSize=9.5, leading=13.5, textColor=INK, spaceAfter=5),
    "bullet": ParagraphStyle("bullet", fontName="Arial", fontSize=9.5, leading=13.5, textColor=INK,
                             leftIndent=12, bulletIndent=2, spaceAfter=3),
    "cell": ParagraphStyle("cell", fontName="Arial", fontSize=8.2, leading=10.2, textColor=INK, alignment=TA_LEFT),
    "cellb": ParagraphStyle("cellb", fontName="Arial-Bold", fontSize=8.2, leading=10.2, textColor=INK),
    "headc": ParagraphStyle("headc", fontName="Arial-Bold", fontSize=8.2, leading=10.2, textColor=colors.white),
    "note": ParagraphStyle("note", fontName="Arial", fontSize=7.8, leading=10, textColor=MUTED, spaceBefore=3, spaceAfter=8),
    "genes": ParagraphStyle("genes", fontName="Arial", fontSize=7.8, leading=10.2, textColor=INK),
}

MODELS = [("logistic_regression", "Regressão Logística"), ("svm", "SVM"), ("random_forest", "Random Forest"),
          ("gradient_boosting", "Gradient Boosting"), ("naive_bayes", "Naive Bayes"), ("knn", "kNN")]
SHORT = {"logistic_regression": "LR", "svm": "SVM", "random_forest": "RF", "gradient_boosting": "GB",
         "naive_bayes": "NB", "knn": "kNN"}
NS = [300, 400, 500, 1000]


def c(text, style="cell"):
    return Paragraph(str(text), st[style])


def pct(x, d=1):
    return f"{x * 100:.{d}f}".replace(".", ",")


def pm(m):
    return f"{pct(m['mean'])} ± {pct(m['std'])}"


# ---------------------------------------------------------------- dados
def count_dir(directory):
    genes, seqs = 0, 0
    for f in glob.glob(os.path.join(directory, "*.fasta")):
        genes += 1
        seqs += sum(1 for line in open(f) if line.startswith(">"))
    return genes, seqs


def tiers_tcc1():
    t = defaultdict(lambda: [0, 0, []])
    for f in sorted(glob.glob(os.path.join(P, "data/raw_negativos/*.fasta"))):
        stem = os.path.basename(f)[:-6]
        gene, cat = stem.split("_", 1)
        tier = "facil" if "housekeeping" in cat else ("dificil" if "nao_ligados" in cat else "intermediario")
        t[tier][0] += 1
        t[tier][1] += sum(1 for line in open(f) if line.startswith(">"))
        t[tier][2].append(gene)
    return t


def tiers_tcc2(directory):
    t = defaultdict(lambda: [0, 0, []])
    for f in sorted(glob.glob(os.path.join(P, directory, "*.fasta"))):
        gene, tier = os.path.basename(f)[:-6].split("_")[:2]
        t[tier][0] += 1
        t[tier][1] += sum(1 for line in open(f) if line.startswith(">"))
        t[tier][2].append(gene)
    return t


def stems(directory):
    return sorted(os.path.basename(f)[:-6] for f in glob.glob(os.path.join(P, directory, "*.fasta")))


def load_results(rdir):
    return {(m, n): json.load(open(os.path.join(P, rdir, f"results_{m}_n{n}.json")))
            for m, _ in MODELS for n in NS}


def load_predictions(path):
    rows = {}
    with open(os.path.join(P, path), encoding="utf-8") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if len(row) == 9 and row[0] in ("INS", "PTPN22", "IL12B", "CARD14", "RUNX3"):
                rows[row[0]] = {"categoria": row[1], "esperado": row[2],
                                "pred": dict(zip([s for _, s in [(m, SHORT[m]) for m, _ in MODELS]], row[3:]))}
    return rows


pos1 = count_dir(os.path.join(P, "data/raw"))
pos2 = count_dir(os.path.join(P, "data/raw_v2"))
neg1 = tiers_tcc1()
neg2 = tiers_tcc2("data/raw_negativos_v2")
blind_pos = count_dir(os.path.join(P, "data/teste_cego_v2"))
blind_neg = tiers_tcc2("data/teste_cego_negativos_v2")
res1 = load_results("reports")
res2 = load_results("reports/v2")
pred1 = load_predictions("reports/tabelas_predicoes.tsv")
pred2 = load_predictions("reports/v2/tabelas_predicoes.tsv")

tot_neg1 = [sum(v[0] for v in neg1.values()), sum(v[1] for v in neg1.values())]
tot_neg2 = [sum(v[0] for v in neg2.values()), sum(v[1] for v in neg2.values())]
bn = [sum(v[0] for v in blind_neg.values()), sum(v[1] for v in blind_neg.values())]
n_diseases = len({os.path.basename(f)[:-6].split("_intermediario_")[1]
                  for f in glob.glob(os.path.join(P, "data/raw_negativos_v2/*_intermediario_*.fasta"))})


def gs(x):
    return f"{x[0]} genes · {x[1]} seqs"


def share(t, total):
    return " / ".join(f"{round(100 * t[k][1] / total)}%" for k in ("facil", "intermediario", "dificil"))


# ---------------------------------------------------------------- tabelas
def base_style(n_rows, zebra_from=1, extra=()):
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), HEAD),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, HEAD),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    return TableStyle(cmds + list(extra))


def table_dataset():
    rows = [[c("Conjunto", "headc"), c("TCC I", "headc"), c("TCC II", "headc")]]
    data = [
        ("Fonte das positivas", "Tsoi et al. (2012) — 21 loci", "Dand et al. (2025) — loci estabelecidos (“Known”)"),
        ("<b>Positivas (treino)</b>", f"<b>{gs(pos1)}</b>", f"<b>{gs(pos2)}</b><br/>RUNX3 e CARD14 reservadas para a validação externa"),
        ("Negativas fáceis<br/>(housekeeping)", gs(neg1["facil"]), gs(neg2["facil"])),
        ("Negativas intermediárias<br/>(outras doenças imunes)", gs(neg1["intermediario"]) + "<br/>Crohn e artrite reumatoide",
         gs(neg2["intermediario"]) + f"<br/>{n_diseases} doenças imunomediadas"),
        ("Negativas difíceis<br/>(imunes não ligadas e parálogos)", gs(neg1["dificil"]) + "<br/>só imunes não ligadas",
         gs(neg2["dificil"]) + "<br/>imunes não ligadas + parálogos dos marcadores"),
        ("<b>Negativas (treino)</b>", f"<b>{gs(tot_neg1)}</b>", f"<b>{gs(tot_neg2)}</b>"),
        ("Proporção fácil / interm. / difícil", share(neg1, tot_neg1[1]), share(neg2, tot_neg2[1])),
        ("<b>Total de treino</b>", f"<b>{pos1[0] + tot_neg1[0]} genes · {pos1[1] + tot_neg1[1]} seqs</b>",
         f"<b>{pos2[0] + tot_neg2[0]} genes · {pos2[1] + tot_neg2[1]} seqs</b>"),
        ("Validação externa", "INS, PTPN22, IL12B, CARD14, RUNX3", "as mesmas 5 proteínas"),
        ("Exclusão de negativas", "STAT4 (locus de psoríase)",
         "genes positivos, genes a &lt;250 kb das 109 regiões de Dand (2025) e genes com associação à psoríase (Open Targets, GWAS Catalog, UniProt)"),
    ]
    for a, b, d in data:
        rows.append([c(a), c(b), c(d)])
    t = Table(rows, colWidths=[4.6 * cm, 5.4 * cm, 7.0 * cm], repeatRows=1)
    extra = [("BACKGROUND", (2, 1), (2, -1), TCC2)]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            extra.append(("BACKGROUND", (0, i), (1, i), ZEBRA))
    t.setStyle(base_style(len(rows), extra=extra))
    return t


def table_cv():
    metrics = [("accuracy", "Acurácia"), ("sensitivity", "Sensibilidade"), ("specificity", "Especificidade"),
               ("roc_auc", "ROC-AUC"), ("mcc", "MCC")]
    rows = [[c("Modelo", "headc"), c("", "headc")] + [c(l, "headc") for _, l in metrics]]
    extra = []
    best1 = max(MODELS, key=lambda m: res1[(m[0], 1000)]["metrics"]["roc_auc"]["mean"])[0]
    best2 = max(MODELS, key=lambda m: res2[(m[0], 1000)]["metrics"]["roc_auc"]["mean"])[0]
    for i, (m, label) in enumerate(MODELS):
        for j, (tag, res, best) in enumerate((("TCC I", res1, best1), ("TCC II", res2, best2))):
            mt = res[(m, 1000)]["metrics"]
            style = "cellb" if m == best else "cell"
            rows.append([c(label if j == 0 else "", "cellb"), c(tag)] + [c(pm(mt[k]), style) for k, _ in metrics])
            r = len(rows) - 1
            if j == 1:
                extra.append(("BACKGROUND", (1, r), (-1, r), TCC2))
        extra.append(("SPAN", (0, len(rows) - 2), (0, len(rows) - 1)))
        extra.append(("LINEBELOW", (0, len(rows) - 1), (-1, len(rows) - 1), 0.6, RULE))
    t = Table(rows, colWidths=[3.0 * cm, 1.4 * cm] + [2.52 * cm] * 5, repeatRows=1)
    t.setStyle(base_style(len(rows), extra=extra))
    return t


def table_auc_by_n():
    rows = [[c("Modelo", "headc"), c("", "headc")] + [c(f"N = {n}", "headc") for n in NS]]
    extra = []
    for m, label in MODELS:
        for j, (tag, res) in enumerate((("TCC I", res1), ("TCC II", res2))):
            vals = [res[(m, n)]["metrics"]["roc_auc"]["mean"] for n in NS]
            top = max(vals)
            rows.append([c(label if j == 0 else "", "cellb"), c(tag)] +
                        [c(pct(v), "cellb" if v == top else "cell") for v in vals])
            if j == 1:
                extra.append(("BACKGROUND", (1, len(rows) - 1), (-1, len(rows) - 1), TCC2))
        extra.append(("SPAN", (0, len(rows) - 2), (0, len(rows) - 1)))
        extra.append(("LINEBELOW", (0, len(rows) - 1), (-1, len(rows) - 1), 0.6, RULE))
    t = Table(rows, colWidths=[3.0 * cm, 1.4 * cm] + [3.15 * cm] * 4, repeatRows=1)
    t.setStyle(base_style(len(rows), extra=extra))
    return t


def fmt_pred(cell):
    sym, val = cell.split(" ", 1)
    color = {"✓": OK, "✗": BAD, "~": WARN}.get(sym, INK)
    glyph = {"✓": "✓", "✗": "✗", "~": "~"}.get(sym, sym)
    return Paragraph(f'<font name="ArialUni" color="{color.hexval()}">{glyph}</font> {val}', st["cell"])


def table_predictions():
    rows = [[c("Proteína", "headc"), c("", "headc"), c("Situação no treino", "headc")] +
            [c(SHORT[m], "headc") for m, _ in MODELS]]
    extra = []
    situ = {
        ("INS", 1): "negativa treinada", ("INS", 2): "negativa não treinada (excluída pelos critérios v2)",
        ("PTPN22", 1): "negativa — apresentada como não treinada, mas a mesma sequência humana estava nos negativos de treino",
        ("PTPN22", 2): "negativa não treinada",
        ("IL12B", 1): "positiva treinada", ("IL12B", 2): "positiva treinada",
        ("CARD14", 1): "positiva não treinada (epitelial)", ("CARD14", 2): "positiva não treinada (epitelial)",
        ("RUNX3", 1): "positiva não treinada (imune sistêmica)", ("RUNX3", 2): "positiva não treinada (imune sistêmica)",
    }
    for prot in ("INS", "PTPN22", "IL12B", "CARD14", "RUNX3"):
        for j, (tag, pred) in enumerate((("TCC I", pred1), ("TCC II", pred2))):
            p = pred[prot]
            name = f"{prot}<br/><font color='{MUTED.hexval()}' size='7'>esperado: {p['esperado'].lower()}</font>"
            rows.append([c(name if j == 0 else "", "cellb"), c(tag), c(situ[(prot, j + 1)])] +
                        [fmt_pred(p["pred"][SHORT[m]]) for m, _ in MODELS])
            if j == 1:
                extra.append(("BACKGROUND", (1, len(rows) - 1), (-1, len(rows) - 1), TCC2))
        extra.append(("SPAN", (0, len(rows) - 2), (0, len(rows) - 1)))
        extra.append(("LINEBELOW", (0, len(rows) - 1), (-1, len(rows) - 1), 0.6, RULE))
    t = Table(rows, colWidths=[2.0 * cm, 1.3 * cm, 4.3 * cm] + [1.57 * cm] * 6, repeatRows=1)
    t.setStyle(base_style(len(rows), extra=extra))
    return t


def table_genes():
    app = [
        ("TCC I — positivas", stems("data/raw")),
        ("TCC I — negativas fáceis", neg1["facil"][2]),
        ("TCC I — negativas intermediárias", neg1["intermediario"][2]),
        ("TCC I — negativas difíceis", neg1["dificil"][2]),
        ("TCC II — positivas (treino)", stems("data/raw_v2")),
        ("TCC II — negativas fáceis (treino)", neg2["facil"][2]),
        ("TCC II — negativas intermediárias (treino)", neg2["intermediario"][2]),
        ("TCC II — negativas difíceis (treino)", neg2["dificil"][2]),
    ]
    rows = [[c("Grupo", "headc"), c("Genes", "headc")]]
    for label, genes in app:
        rows.append([c(f"<b>{label}</b><br/><font color='{MUTED.hexval()}'>{len(genes)} genes</font>"),
                     Paragraph(", ".join(sorted(genes)), st["genes"])])
    t = Table(rows, colWidths=[4.2 * cm, 12.8 * cm], repeatRows=1)
    extra = [("VALIGN", (0, 1), (-1, -1), "TOP")]
    for i in range(1, len(rows)):
        if app[i - 1][0].startswith("TCC II"):
            extra.append(("BACKGROUND", (0, i), (-1, i), TCC2))
    t.setStyle(base_style(len(rows), extra=extra))
    return t


# ---------------------------------------------------------------- texto
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Arial", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, "TCC — Classificação de proteínas associadas à psoríase · Comparação TCC I × TCC II")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
    canvas.restoreState()


def main():
    story = [
        Paragraph("TCC I × TCC II: dados e resultados", st["title"]),
        Paragraph("Comparação entre o dataset e os resultados do TCC I e a primeira rodada do TCC II "
                  "(mesma pipeline e mesmos grids de hiperparâmetros do TCC I). Rodada de 20/10/2026.", st["sub"]),
        Paragraph("1. Proteínas em cada grupo", st["h1"]),
        table_dataset(),
        Paragraph("seqs = sequências. No TCC I, os genes de Crohn e "
                  "artrite reumatoide estavam no grupo “difícil”; aqui foram reagrupados como intermediários para "
                  "seguir os níveis do TCC II.", st["note"]),
        PageBreak(),
        Paragraph("2. Resultados da validação cruzada (N = 1000)", st["h1"]),
        Paragraph("Média ± desvio padrão nos 10 folds, em %, para os hiperparâmetros vencedores do GridSearchCV. "
                  "Em negrito, o melhor modelo de cada dataset por ROC-AUC.", st["body"]),
        table_cv(),
        Spacer(1, 6),
        KeepTogether([
            Paragraph("3. ROC-AUC por tamanho do vetor", st["h1"]),
            Paragraph("Média nos 10 folds, em %. Em negrito, o melhor N de cada modelo.", st["body"]),
            table_auc_by_n(),
        ]),
        PageBreak(),
        Paragraph("4. Validação externa (5 proteínas)", st["h1"]),
        Paragraph("Classificação de cada proteína pelos modelos finais (N = 1000). O percentual é a confiança na classe "
                  "prevista: “<font name='ArialUni'>✗</font> 69,7%” significa que o modelo errou com 69,7% de confiança.", st["body"]),
        table_predictions(),
        Paragraph('<font name="ArialUni">✓</font> correto · <font name="ArialUni">✗</font> incorreto · '
                  "~ correto, mas com confiança abaixo de 65%.", st["note"]),
        PageBreak(),
        Paragraph("Apêndice — genes de cada grupo", st["h1"]),
        table_genes(),
        Paragraph("Validação externa (as mesmas nos dois): INS, PTPN22, IL12B, CARD14, RUNX3.", st["note"]),
    ]
    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, title="TCC I × TCC II — dados e resultados",
                            author="Grupo do TCC", subject="Comparação TCC I × TCC II")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    log.info("PDF: %s", OUT)
    log.info("positivas %s / %s · negativas %s / %s · teste cego %s / %s", pos1, pos2, tot_neg1, tot_neg2, blind_pos, bn)


if __name__ == "__main__":
    main()
