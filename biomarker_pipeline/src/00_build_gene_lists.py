"""
Módulo 00a — Listas de genes do TCC II a partir de Dand et al. (2025), Supplementary Data 20.

Fonte: "Summary of psoriasis-associated regions, independent signals and implicated genes"
(Nat Commun 16:2051, 2025; doi:10.1038/s41467-025-56719-8, arquivo MOESM3).

A coluna "Established psoriasis loci in Europeans" particiona as 109 regiões:
  - preenchida   → Known          (59) → positivos de TREINO
  - "None"       → Newly reported (50) → positivos do TESTE CEGO

Treino (Known): genes da coluna "Previously proposed" (nota f — Stuart, Tsoi, Hambro &
Elder, 2018), o mesmo critério dos 28 genes do TCC I, com as decisões editoriais:
  1. remove tokens não codificantes (NONCODING_TOKENS)
  2. expande compostos (COMPOSITES)
  3. descarta genes propostos só para artrite psoriásica — "(PsA)" sem contraparte

Teste cego (Newly reported): a SD20 não traz "Previously proposed" para essas regiões,
então o gene de cada locus é escolhido pela evidência mais forte disponível:
  A — variante que altera proteína (PAV) ou alvo terapêutico existente
  B — candidato TWAS
  C — sinal dentro do corpo do gene ("within")
  D — gene codificador mais próximo de sinal intergênico (evidência fraca)
Regiões em LD com um locus de treino são marcadas excluir_validacao=sim.

RUNX3 e CARD14 são Known, mas ficam fora do treino (TCC1_VALIDATION_HOLDOUT): foram as
positivas "não treinadas" da validação externa do TCC I e seguem assim para comparação.

Saídas (data/gene_lists/):
  dand2025_treino_estabelecidos.tsv              — 59 regiões Known e genes extraídos
  dand2025_validacao_ineditos.tsv                — 50 regiões Newly reported, gene escolhido e nível
  positivos_dand2025_estabelecidos_tierA.txt     — genes de treino (entrada de 00_fetch_uniprot.py)
  teste_cego_dand2025_ineditos.txt               — genes do teste cego (entrada de 00_fetch_uniprot.py)

Uso:
    python src/00_build_gene_lists.py
"""

import os
import re
import sys
import csv
import logging

import openpyxl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

GENE_LIST_DIR  = "data/gene_lists"
XLSX_PATH      = os.path.join(GENE_LIST_DIR, "dand2025_supplementary_data20.xlsx")
SHEET_NAME     = "Supplementary Data 20"
TRAIN_TSV      = os.path.join(GENE_LIST_DIR, "dand2025_treino_estabelecidos.tsv")
BLIND_TSV      = os.path.join(GENE_LIST_DIR, "dand2025_validacao_ineditos.tsv")
TRAIN_TXT      = os.path.join(GENE_LIST_DIR, "positivos_dand2025_estabelecidos_tierA.txt")
BLIND_TXT      = os.path.join(GENE_LIST_DIR, "teste_cego_dand2025_ineditos.txt")

# Colunas da SD20 (índice 0 = coluna A, vazia)
COL_REGION, COL_CHR, COL_LD, COL_ESTABLISHED = 1, 2, 3, 6
COL_PAV, COL_TWAS, COL_PROPOSED, COL_THERAPY, COL_CLOSEST = 10, 11, 12, 13, 14
EMPTY_VALUES   = {None, "", ".", "-", "None"}
NO_ESTABLISHED = "None"

PSA_MARK = "(PsA)"
NONCODING_TOKENS = {"MHC", "HLA", "LINC00330", "LINC01885", "SNORD74", "RN7SKP9",
                    "RP11-61O1.1", "LCE3D-LCE3E"}
COMPOSITES = {
    "LCE3C/B":         ["LCE3C", "LCE3B"],
    "STAT5A/B":        ["STAT5A", "STAT5B"],
    "RNF145 (IL12B)":  ["RNF145"],
}
NONCODING_RE = re.compile(r"^(RP\d+-|LINC\d|SNORD|RN7SK|MIR\d)")

# Positivas da validação externa do TCC I (data/input/) que são Known na SD20: ficam FORA
# do treino para comparar com o TCC I nas mesmas condições ("positiva não treinada")
TCC1_VALIDATION_HOLDOUT = {
    "RUNX3":  "validação externa do TCC I (via imune sistêmica)",
    "CARD14": "validação externa do TCC I (via epitelial)",
}

# Regiões Newly reported em LD com um locus de treino: região → motivo
BLIND_EXCLUDE = {
    28: "bloco de LD adjacente à região 29 (Known); TP63, único gene codificador a <250 kb, "
        "já está no treino",
}

SOURCE_NOTE = ("Fonte: Dand et al. (2025), Nat Commun 16:2051, Supplementary Data 20 "
               "(dand2025_supplementary_data20.xlsx). Gerado por src/00_build_gene_lists.py.")


def is_empty(value) -> bool:
    return value is None or str(value).strip() in EMPTY_VALUES


def split_genes(cell) -> list[str]:
    if is_empty(cell):
        return []
    return [g.strip() for g in str(cell).split(",") if g.strip() and g.strip() not in EMPTY_VALUES]


def load_regions() -> list[dict]:
    if not os.path.exists(XLSX_PATH):
        log.error(f"{XLSX_PATH} não encontrado. Baixe o MOESM3 de doi:10.1038/s41467-025-56719-8 "
                  f"(static-content.springer.com) e salve com esse nome.")
        sys.exit(1)
    ws = openpyxl.load_workbook(XLSX_PATH, read_only=True)[SHEET_NAME]
    regions = []
    for row in ws.iter_rows(values_only=True):
        if len(row) <= COL_CLOSEST or not isinstance(row[COL_REGION], int):
            continue
        established = str(row[COL_ESTABLISHED]).strip()
        regions.append({
            "regiao":       row[COL_REGION],
            "chr":          row[COL_CHR],
            "bloco_ld":     row[COL_LD],
            # A coluna Type é uma fórmula =IF(G="None",...) — recalculada aqui
            "tipo":         "Newly reported" if established == NO_ESTABLISHED else "Known",
            "estabelecido": established.replace("\n", "; "),
            "pav":          split_genes(row[COL_PAV]),
            "twas":         split_genes(row[COL_TWAS]),
            "propostos":    split_genes(row[COL_PROPOSED]),
            "terapia":      "" if is_empty(row[COL_THERAPY]) else str(row[COL_THERAPY]).strip(),
            "mais_proximo": "" if is_empty(row[COL_CLOSEST]) else str(row[COL_CLOSEST]).strip(),
        })
    return regions


def proposed_to_genes(tokens: list[str]) -> tuple[list[str], list[str]]:
    """Aplica as decisões editoriais; devolve (genes incluídos, tokens descartados com motivo)."""
    psoriasis, psa_only, dropped = [], [], []
    for tok in tokens:
        if tok in COMPOSITES:
            psoriasis.extend(COMPOSITES[tok])
        elif tok.endswith(PSA_MARK):
            psa_only.append(tok[: -len(PSA_MARK)].strip())
        else:
            psoriasis.append(tok)
    for gene in psa_only:
        if gene not in psoriasis and gene not in NONCODING_TOKENS:
            dropped.append(f"{gene} (só PsA)")
    genes = []
    for gene in psoriasis:
        if gene in NONCODING_TOKENS or NONCODING_RE.match(gene):
            dropped.append(f"{gene} (não codificante)")
        elif gene in TCC1_VALIDATION_HOLDOUT:
            dropped.append(f"{gene} (reservado: {TCC1_VALIDATION_HOLDOUT[gene]})")
        elif gene not in genes:
            genes.append(gene)
    return genes, dropped


def closest_gene(text: str) -> tuple[str, str]:
    """Do texto 'Closest protein-coding' devolve (gene, nível C/D) para o sinal 1."""
    # Um sinal por linha ("Signal 1: ...\nSignal 2: ..."); vizinhos separados por ";"
    first_signal = re.sub(r"^Signal \d+:\s*", "", text.splitlines()[0].strip())
    within = re.match(r"(\S+) \(within\)", first_signal)
    if within:
        return within.group(1), "C"
    flanks = re.findall(r"(\S+) \(([+-]\d+) KB\)", first_signal)
    if not flanks:
        return "", ""
    gene, _ = min(flanks, key=lambda f: abs(int(f[1])))
    return gene, "D"


def therapy_gene(text: str) -> str:
    return text.split(" ")[0] if text else ""


def choose_blind_gene(region: dict) -> tuple[list[str], str, str]:
    """Escolhe o(s) gene(s) do locus inédito pela evidência mais forte (A > B > C > D)."""
    if region["pav"] or region["terapia"]:
        genes = list(region["pav"])
        target = therapy_gene(region["terapia"])
        if target and target not in genes:
            genes.append(target)
        return genes, "A", "PAV" if region["pav"] else "alvo terapêutico"
    twas = [g for g in region["twas"] if not NONCODING_RE.match(g)]
    gene, level = closest_gene(region["mais_proximo"])
    if twas:
        # Vários candidatos TWAS: fica o que também é o gene mais próximo, se houver
        return ([gene] if gene in twas else twas), "B", "TWAS"
    if gene:
        return [gene], level, "dentro do gene" if level == "C" else "mais próximo (intergênico)"
    return [], "", "sem gene codificador"


def write_tsv(path: str, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_gene_txt(path: str, title: str, genes: list[str], notes: list[str]) -> None:
    with open(path, "w") as fh:
        fh.write(f"# {title}\n# {SOURCE_NOTE}\n")
        for note in notes:
            fh.write(f"# {note}\n")
        fh.write(f"# {len(genes)} genes\n")
        fh.writelines(f"{g}\n" for g in genes)


def main():
    regions = load_regions()
    known = [r for r in regions if r["tipo"] == "Known"]
    newly = [r for r in regions if r["tipo"] == "Newly reported"]
    log.info(f"{len(regions)} regiões: {len(known)} Known, {len(newly)} Newly reported")

    train_genes, train_rows = [], []
    for r in known:
        genes, dropped = proposed_to_genes(r["propostos"])
        if not genes:
            log.warning(f"Região {r['regiao']}: nenhum gene codificador ({', '.join(r['propostos']) or '—'})")
        train_genes.extend(g for g in genes if g not in train_genes)
        train_rows.append([r["regiao"], r["chr"], r["bloco_ld"], r["estabelecido"],
                           ", ".join(r["propostos"]), ", ".join(genes), "; ".join(dropped)])
    write_tsv(TRAIN_TSV, ["regiao", "chr", "bloco_ld", "loci_estabelecidos", "propostos_sd20",
                          "genes_incluidos", "descartados"], train_rows)

    blind_genes, blind_rows = [], []
    for r in newly:
        genes, level, basis = choose_blind_gene(r)
        overlap = [g for g in genes if g in train_genes]
        reason = BLIND_EXCLUDE.get(r["regiao"], "")
        if overlap and not reason:
            reason = f"gene já no treino: {', '.join(overlap)}"
        if not genes and not reason:
            reason = "sem gene codificador"
        if not reason:
            blind_genes.extend(g for g in genes if g not in blind_genes)
        blind_rows.append([r["regiao"], r["chr"], r["bloco_ld"], ", ".join(r["pav"]),
                           ", ".join(r["twas"]), r["terapia"], r["mais_proximo"].replace("\n", "; "),
                           ", ".join(genes), level, basis,
                           "sim" if reason else "nao", reason])
    write_tsv(BLIND_TSV, ["regiao", "chr", "bloco_ld", "pav", "twas", "alvo_terapeutico",
                          "mais_proximo", "genes_escolhidos", "nivel_evidencia", "base",
                          "excluir_validacao", "motivo"], blind_rows)

    write_gene_txt(TRAIN_TXT, "Positivos de TREINO — regiões Known, genes 'Previously proposed' (tier A)",
                   train_genes,
                   [f"Removidos (não codificantes): {', '.join(sorted(NONCODING_TOKENS))}",
                    "Expandidos: " + "; ".join(f"{k} → {'+'.join(v)}" for k, v in COMPOSITES.items()),
                    "Descartados genes marcados (PsA) sem contraparte de psoríase",
                    "Reservados para comparar com a validação externa do TCC I: "
                    + ", ".join(TCC1_VALIDATION_HOLDOUT)])
    write_gene_txt(BLIND_TXT, "Positivos do TESTE CEGO — regiões Newly reported",
                   blind_genes,
                   ["Nível de evidência por região em dand2025_validacao_ineditos.tsv "
                    "(A=PAV/alvo terapêutico, B=TWAS, C=dentro do gene, D=mais próximo)"])

    levels = {}
    for row in blind_rows:
        if row[10] == "nao":
            levels[row[8]] = levels.get(row[8], 0) + 1
    log.info(f"Treino: {len(train_genes)} genes de {sum(1 for r in train_rows if r[5])} regiões → {TRAIN_TXT}")
    log.info(f"Teste cego: {len(blind_genes)} genes de {sum(levels.values())} regiões "
             f"(níveis: {dict(sorted(levels.items()))}) → {BLIND_TXT}")
    for row in blind_rows:
        if row[10] == "sim":
            log.info(f"  Região {row[0]} excluída do teste cego: {row[11]}")


if __name__ == "__main__":
    main()
