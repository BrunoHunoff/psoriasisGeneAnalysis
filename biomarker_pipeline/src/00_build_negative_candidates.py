"""
Módulo 00c — Lista de genes candidatos a NEGATIVOS do TCC II, por nível de dificuldade.

Fontes de candidatos:
  - fácil (housekeeping):      18 housekeeping do TCC I + amostra aleatória (RANDOM_STATE)
                               de Eisenberg & Levanon (2013), "Human housekeeping genes, revisited"
  - intermediário:             genes de outras doenças imunomediadas — TCC I (Crohn, artrite
                               reumatoide) + 00_fetch_negatives.py (Open Targets)
  - difícil:                   imunes sem ligação com psoríase (TCC I + citocinas/receptores)
                               e parálogos dos biomarcadores (00_fetch_negatives.py)

Exclusões (qualquer uma exclui o gene, em todos os níveis):
  1. positivo em qualquer conjunto: TCC I (data/raw), treino v2, teste cego v2, holdouts
  2. qualquer gene codificador a <250 kb de QUALQUER das 109 regiões de Dand et al. (2025)
     (SD20, coluna "All protein-coding within 250kb") — Known e Newly reported
  3. reports/negativos/psoriase_exclusao.tsv (Open Targets / GWAS Catalog / UniProt, gerado
     por 00_fetch_negatives.py)
  4. PTPN22 — negativa "não treinada" da validação externa do TCC I

Saídas (data/gene_lists/):
  negativos_candidatos_v2.tsv   — gene, nivel, subcategoria, origem
  negativos_candidatos_v2.txt   — entrada de 00_fetch_uniprot.py
  negativos_excluidos_v2.tsv    — candidatos descartados e motivo

Uso:
    python src/00_build_negative_candidates.py
    python src/00_fetch_uniprot.py --genes data/gene_lists/negativos_candidatos_v2.txt --out-dir data/negativos_pool_v2 --max-seqs 4
"""

import os
import re
import sys
import csv
import random
import logging

import openpyxl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

RANDOM_STATE   = 42
GENE_LIST_DIR  = "data/gene_lists"
RAW_POS_DIR    = "data/raw"
RAW_NEG_DIR    = "data/raw_negativos"
IGOR_MANIFEST  = os.path.join("reports", "negativos", "negativos_manifest.tsv")
PSO_EXCLUSION  = os.path.join("reports", "negativos", "psoriase_exclusao.tsv")
HK_LIST        = os.path.join(GENE_LIST_DIR, "eisenberg2013_housekeeping.txt")
SD20_XLSX      = os.path.join(GENE_LIST_DIR, "dand2025_supplementary_data20.xlsx")
SD20_SHEET     = "Supplementary Data 20"
SD20_COL_ALL_250KB = 15
POSITIVE_LISTS = [os.path.join(GENE_LIST_DIR, "positivos_dand2025_estabelecidos_tierA.txt"),
                  os.path.join(GENE_LIST_DIR, "teste_cego_dand2025_ineditos.txt")]
OUT_TSV        = os.path.join(GENE_LIST_DIR, "negativos_candidatos_v2.tsv")
OUT_TXT        = os.path.join(GENE_LIST_DIR, "negativos_candidatos_v2.txt")
OUT_EXCLUDED   = os.path.join(GENE_LIST_DIR, "negativos_excluidos_v2.tsv")

# Genes positivos fora das listas: holdouts da validação do TCC I e símbolo antigo do TCC I
EXTRA_POSITIVES  = {"RUNX3", "CARD14", "IL28RA", "IFNLR1"}
# Negativa "não treinada" da validação externa do TCC I (data/input/)
NEGATIVE_HOLDOUT = {"PTPN22": "validação externa do TCC I (negativa não treinada)"}

# Candidatos housekeeping amostrados de Eisenberg & Levanon (2013); folga sobre o necessário
HK_SAMPLE_N = 60

NIVEL_FACIL, NIVEL_INTERM, NIVEL_DIFICIL = "facil", "intermediario", "dificil"
HK_CATEGORY      = "housekeeping_facil"
IMMUNE_CATEGORY  = "nao_ligados_psioriase_dificil"
PARALOG_PREFIX   = "paralogo_"
FASTA_EXTS       = (".fasta", ".fa")


def read_gene_txt(path: str) -> set[str]:
    with open(path) as fh:
        return {l.split("#")[0].strip().upper() for l in fh if l.split("#")[0].strip()}


def dand_locus_genes() -> set[str]:
    """Todos os genes codificadores a <250 kb das 109 regiões da SD20."""
    ws = openpyxl.load_workbook(SD20_XLSX, read_only=True)[SD20_SHEET]
    genes = set()
    for row in ws.iter_rows(values_only=True):
        if len(row) > SD20_COL_ALL_250KB and isinstance(row[1], int) and row[SD20_COL_ALL_250KB]:
            for inside in re.findall(r"\(([^)]*)\)", str(row[SD20_COL_ALL_250KB])):
                genes.update(g.strip().upper() for g in inside.split(",") if g.strip())
    return genes


def classify(category: str) -> tuple[str, str]:
    """Categoria do nome de arquivo/manifesto → (nível, subcategoria)."""
    if category == HK_CATEGORY:
        return NIVEL_FACIL, "housekeeping"
    if category == IMMUNE_CATEGORY:
        return NIVEL_DIFICIL, "imune_nao_psoriase"
    if category.startswith(PARALOG_PREFIX):
        return NIVEL_DIFICIL, "paralogo"
    return NIVEL_INTERM, category   # nome da doença (Crohn, Lupus, ...)


def tcc1_candidates(igor_files: set[str]) -> list[dict]:
    out = []
    for fname in sorted(os.listdir(RAW_NEG_DIR)):
        if not fname.endswith(FASTA_EXTS) or fname in igor_files:
            continue
        gene, category = os.path.splitext(fname)[0].split("_", 1)
        nivel, sub = classify(category)
        out.append({"gene": gene.upper(), "nivel": nivel, "subcategoria": sub, "origem": "TCC I"})
    return out


def igor_candidates() -> tuple[list[dict], set[str]]:
    out, files = [], set()
    with open(IGOR_MANIFEST) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            files.add(r["arquivo"])
            nivel, sub = classify(r["categoria"])
            out.append({"gene": r["gene"].upper(), "nivel": nivel, "subcategoria": sub,
                        "origem": f"00_fetch_negatives ({r['fonte']})"})
    return out, files


def main():
    for path in [SD20_XLSX, HK_LIST, IGOR_MANIFEST, PSO_EXCLUSION, *POSITIVE_LISTS]:
        if not os.path.exists(path):
            log.error(f"{path} não encontrado. Rode antes 00_build_gene_lists.py / 00_fetch_negatives.py.")
            sys.exit(1)

    positives = set(EXTRA_POSITIVES)
    for path in POSITIVE_LISTS:
        positives |= read_gene_txt(path)
    positives |= {f.split(".")[0].upper() for f in os.listdir(RAW_POS_DIR) if f.endswith(FASTA_EXTS)}
    locus_genes = dand_locus_genes()
    with open(PSO_EXCLUSION) as fh:
        pso_excluded = {r["gene"].upper(): r["motivos"] for r in csv.DictReader(fh, delimiter="\t")}
    log.info(f"Bloqueio: {len(positives)} positivos, {len(locus_genes)} genes em loci da Dand 2025, "
             f"{len(pso_excluded)} com associação à psoríase")

    def exclusion_reason(gene: str) -> str:
        if gene in positives:
            return "positivo (TCC I ou v2)"
        if gene in locus_genes:
            return "a <250 kb de uma região de psoríase (Dand 2025)"
        if gene in pso_excluded:
            return f"associação com psoríase: {pso_excluded[gene]}"
        if gene in NEGATIVE_HOLDOUT:
            return NEGATIVE_HOLDOUT[gene]
        return ""

    igor, igor_files = igor_candidates()
    pool = tcc1_candidates(igor_files) + igor

    with open(HK_LIST) as fh:
        hk_all = sorted({l.split()[0].upper() for l in fh if l.strip()})
    taken = {c["gene"] for c in pool}
    hk_eligible = [g for g in hk_all if g not in taken and not exclusion_reason(g)]
    hk_sample = random.Random(RANDOM_STATE).sample(hk_eligible, HK_SAMPLE_N)
    pool += [{"gene": g, "nivel": NIVEL_FACIL, "subcategoria": "housekeeping",
              "origem": "Eisenberg & Levanon (2013)"} for g in sorted(hk_sample)]
    log.info(f"Housekeeping: {len(hk_eligible)} elegíveis de {len(hk_all)}; amostrados {HK_SAMPLE_N}")

    kept, excluded, seen = [], [], set()
    for c in pool:
        if c["gene"] in seen:
            continue
        seen.add(c["gene"])
        reason = exclusion_reason(c["gene"])
        (excluded if reason else kept).append({**c, "motivo": reason})

    fields = ["gene", "nivel", "subcategoria", "origem"]
    with open(OUT_TSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(kept)
    with open(OUT_EXCLUDED, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields + ["motivo"], delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(excluded)
    with open(OUT_TXT, "w") as fh:
        fh.write("# Candidatos a negativos do TCC II — gerado por src/00_build_negative_candidates.py\n")
        fh.writelines(f"{c['gene']}\n" for c in kept)

    by_level = {}
    for c in kept:
        by_level[c["nivel"]] = by_level.get(c["nivel"], 0) + 1
    log.info(f"Candidatos: {len(kept)} genes {by_level} → {OUT_TSV}")
    for c in excluded:
        log.info(f"  Excluído {c['gene']} ({c['nivel']}, {c['origem']}): {c['motivo'][:90]}")


if __name__ == "__main__":
    main()
