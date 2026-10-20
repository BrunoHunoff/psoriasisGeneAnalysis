"""
Módulo 07 — Predição de novas proteínas.

Coloque um ou mais arquivos FASTA em data/input/ e rode:

    python src/07_predict.py --model svm --dataset tcc2
    python src/07_predict.py --model svm --dataset tcc2 --k 4 --n-components 4000

Para cada sequência encontrada, o script imprime se ela é um candidato
a biomarcador da psoríase (POSITIVO) ou não (NEGATIVO), com a confiança
da predição. A sequência é vetorizada com a mesma representação e a mesma
projeção do treino (refeita a partir de representação, N e seed — common.kmer_vectors).

Pré-requisito: rode antes, com a mesma representação e o mesmo --n-components,
    python src/06_train_final.py --model svm --dataset tcc2
"""

import os
import sys
import json
import pickle
import logging
import argparse

from Bio import SeqIO

from common import (FASTA_EXTS, INPUT_DIR, add_dataset_arg, add_representation_args,
                    classifier_path, dataset_paths, patterns_from_args, rep_name,
                    sequences_to_vectors, vector_dim)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prediz se proteínas em data/input/ são biomarcadores da psoríase."
    )
    parser.add_argument(
        "--model", required=True,
        help="Nome do modelo treinado (ex: svm, random_forest, logistic_regression).",
    )
    parser.add_argument(
        "--fasta", default=None,
        help="Arquivo FASTA específico (opcional). Se omitido, processa todos em data/input/.",
    )
    add_dataset_arg(parser)
    add_representation_args(parser)
    return parser.parse_args()


def load_fasta_files(input_dir: str, specific: str | None) -> list[tuple[str, str]]:
    """Retorna lista de (seq_id, sequência) a partir do diretório ou arquivo específico."""
    if specific:
        paths = [specific]
    else:
        if not os.path.isdir(input_dir):
            log.error("Pasta '%s' não encontrada. Crie-a e coloque os FASTAs lá.", input_dir)
            sys.exit(1)
        paths = [
            os.path.join(input_dir, f)
            for f in sorted(os.listdir(input_dir))
            if os.path.splitext(f)[1].lower() in FASTA_EXTS
        ]

    if not paths:
        log.error("Nenhum arquivo .fasta encontrado. Adicione arquivos em '%s'.", input_dir)
        sys.exit(1)

    records = []
    for path in paths:
        if not os.path.exists(path):
            log.error("Arquivo não encontrado: %s", path)
            sys.exit(1)
        seqs = list(SeqIO.parse(path, "fasta"))
        log.info("Arquivo '%s': %d sequência(s)", os.path.basename(path), len(seqs))
        for rec in seqs:
            records.append((rec.id, str(rec.seq).upper()))
    return records


def main():
    args = parse_args()
    os.makedirs(INPUT_DIR, exist_ok=True)
    patterns = patterns_from_args(args)
    n_components = vector_dim(patterns, args.n_components)
    models_dir = dataset_paths(args.dataset, patterns)["models_dir"]

    clf_path = classifier_path(models_dir, args.model, n_components)
    if not os.path.exists(clf_path):
        log.error(
            "Classificador não encontrado: %s\n"
            "  → rode primeiro: python src/06_train_final.py --model %s --dataset %s "
            "(com a mesma representação e o mesmo --n-components)",
            clf_path, args.model, args.dataset,
        )
        sys.exit(1)
    with open(clf_path, "rb") as f:
        clf = pickle.load(f)
    log.info("Modelo '%s' carregado. Representação %s, N=%d.", args.model, rep_name(patterns), n_components)

    records = load_fasta_files(INPUT_DIR, args.fasta)
    log.info("Total de sequências a predizer: %d\n", len(records))
    X_all, kept = sequences_to_vectors([seq for _, seq in records], patterns, args.n_components)
    rows = {i: row for row, i in enumerate(kept)}

    print("=" * 65)
    print(f"  RESULTADOS — modelo: {args.model}  |  dataset: {args.dataset}  |  "
          f"{rep_name(patterns)}  |  N={n_components}")
    print("=" * 65)

    for i, (seq_id, seq) in enumerate(records):
        if i not in rows:
            print(f"  {'[IGNORADA]':<12}  {seq_id}  (sequência muito curta)")
            continue

        X = X_all[rows[i]].reshape(1, -1)
        pred  = clf.predict(X)[0]
        proba = clf.predict_proba(X)[0]
        confidence = proba[pred] * 100

        label    = "POSITIVO ✓" if pred == 1 else "NEGATIVO  "
        bar_len  = int(confidence / 5)
        bar      = "█" * bar_len + "░" * (20 - bar_len)

        print(f"\n  {label}  |  confiança: {confidence:5.1f}%  [{bar}]")
        print(f"  Sequência: {seq_id}")
        print(f"  P(biomarcador psoríase): {proba[1]*100:.1f}%   P(negativo): {proba[0]*100:.1f}%")

    print("\n" + "=" * 65)
    print("  POSITIVO = candidato a biomarcador da psoríase")
    print("  NEGATIVO = não identificado como biomarcador")
    print("=" * 65)


if __name__ == "__main__":
    main()
