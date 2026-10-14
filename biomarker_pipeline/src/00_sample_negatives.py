"""
Módulo 00d — Amostragem sistemática dos NEGATIVOS do TCC II (treino e teste cego).

Parte do pool coletado por 00_fetch_uniprot.py (data/negativos_pool_v2/) e da lista de
candidatos com nível de dificuldade (00_build_negative_candidates.py) e monta:

  - data/raw_negativos_v2/         negativos de TREINO      (n seqs = positivos de data/raw_v2)
  - data/teste_cego_negativos_v2/  negativos do TESTE CEGO  (n seqs = positivos de data/teste_cego_v2)

Regras:
  1. Proporção 1:1 com os positivos de cada conjunto, contada em SEQUÊNCIAS.
  2. Dentro de cada conjunto, ⅓ para cada nível (fácil / intermediário / difícil).
  3. A amostragem é por GENE: um gene (e todos os seus homólogos) cai em um conjunto só.
  4. Dentro de cada nível, as subcategorias (doença, parálogo, imune...) se alternam em
     rodízio, cada uma embaralhada com RANDOM_STATE — evita que uma subcategoria domine.
  5. Para fechar o número exato, o último gene de cada nível pode entrar com parte das
     sequências; o FASTA já vem ordenado por prioridade de espécie (humano primeiro).

Saídas:
  {conjunto}/{GENE}_{nivel}_{subcategoria}.fasta
  reports/coleta/amostragem_negativos_v2.tsv — gene, nível, subcategoria, conjunto, n_seqs

Uso:
    python src/00_sample_negatives.py
"""

import os
import sys
import csv
import shutil
import random
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

RANDOM_STATE   = 42
CANDIDATES_TSV = "data/gene_lists/negativos_candidatos_v2.tsv"
POOL_DIR       = "data/negativos_pool_v2"
SETS = [  # (nome, diretório dos positivos, diretório de saída dos negativos) — teste cego primeiro
    ("teste_cego", "data/teste_cego_v2", "data/teste_cego_negativos_v2"),
    ("treino",     "data/raw_v2",        "data/raw_negativos_v2"),
]
LEVELS      = ["facil", "intermediario", "dificil"]
REPORT_PATH = os.path.join("reports", "coleta", "amostragem_negativos_v2.tsv")
FASTA_EXT   = ".fasta"


def read_fasta_records(path: str) -> list[str]:
    with open(path) as fh:
        return [">" + rec for rec in fh.read().split(">")[1:]]


def count_seqs(directory: str) -> int:
    return sum(len(read_fasta_records(os.path.join(directory, f)))
               for f in os.listdir(directory) if f.endswith(FASTA_EXT))


def level_targets(total: int) -> dict[str, int]:
    """Divide total em ⅓ por nível; o resto vai para os primeiros níveis."""
    base, rest = divmod(total, len(LEVELS))
    return {lvl: base + (1 if i < rest else 0) for i, lvl in enumerate(LEVELS)}


def round_robin_order(candidates: list[dict], rng: random.Random) -> list[dict]:
    """Embaralha cada subcategoria e intercala as subcategorias em rodízio."""
    by_sub: dict[str, list[dict]] = {}
    for c in sorted(candidates, key=lambda c: c["gene"]):
        by_sub.setdefault(c["subcategoria"], []).append(c)
    queues = []
    for sub in sorted(by_sub):
        genes = by_sub[sub]
        rng.shuffle(genes)
        queues.append(genes)
    order = []
    while any(queues):
        for q in queues:
            if q:
                order.append(q.pop(0))
    return order


def main():
    for path in [CANDIDATES_TSV, POOL_DIR] + [p for _, p, _ in SETS]:
        if not os.path.exists(path):
            log.error(f"{path} não encontrado. Rode antes 00_build_negative_candidates.py e "
                      f"00_fetch_uniprot.py (positivos e pool de negativos).")
            sys.exit(1)

    with open(CANDIDATES_TSV) as fh:
        candidates = [c for c in csv.DictReader(fh, delimiter="\t")
                      if os.path.exists(os.path.join(POOL_DIR, c["gene"] + FASTA_EXT))]
    for c in candidates:
        c["records"] = read_fasta_records(os.path.join(POOL_DIR, c["gene"] + FASTA_EXT))

    rng = random.Random(RANDOM_STATE)
    queues = {lvl: round_robin_order([c for c in candidates if c["nivel"] == lvl], rng)
              for lvl in LEVELS}
    for lvl in LEVELS:
        log.info(f"Pool {lvl}: {len(queues[lvl])} genes, "
                 f"{sum(len(c['records']) for c in queues[lvl])} seqs")

    report_rows = []
    for set_name, pos_dir, out_dir in SETS:
        n_pos = count_seqs(pos_dir)
        targets = level_targets(n_pos)
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        for lvl in LEVELS:
            taken = 0
            while taken < targets[lvl]:
                if not queues[lvl]:
                    log.error(f"Pool {lvl} esgotado em {set_name} ({taken}/{targets[lvl]} seqs). "
                              f"Aumente os candidatos em 00_build_negative_candidates.py.")
                    sys.exit(1)
                c = queues[lvl].pop(0)
                records = c["records"][: targets[lvl] - taken]
                fname = f"{c['gene']}_{lvl}_{c['subcategoria']}{FASTA_EXT}"
                with open(os.path.join(out_dir, fname), "w") as fh:
                    fh.writelines(records)
                taken += len(records)
                report_rows.append([c["gene"], lvl, c["subcategoria"], c["origem"], set_name,
                                    len(records), len(c["records"])])
        log.info(f"{set_name}: {n_pos} positivos → {sum(targets.values())} negativos "
                 f"{targets} → {out_dir}")

    with open(REPORT_PATH, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["gene", "nivel", "subcategoria", "origem", "conjunto", "n_seqs_usadas", "n_seqs_pool"])
        w.writerows(report_rows)
    log.info(f"Relatório: {REPORT_PATH}")


if __name__ == "__main__":
    main()
