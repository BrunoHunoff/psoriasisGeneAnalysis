"""
Módulo 07 — Predição de novas proteínas.

Coloque um ou mais arquivos FASTA em data/input/ e rode:

    python src/07_predict.py --model svm --dataset tcc2

Para cada sequência encontrada, o script imprime se ela é um candidato
a biomarcador da psoríase (POSITIVO) ou não (NEGATIVO), com a confiança
da predição.

Pré-requisito: rode antes
    python src/06_train_final.py --model svm --dataset tcc2
"""

import os
import sys
import json
import pickle
import logging
import argparse

from Bio import SeqIO

from common import (FASTA_EXTS, INPUT_DIR, add_dataset_arg, build_vocab, classifier_path,
                    dataset_paths, projector_path, sequence_to_vector)

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
    models_dir = dataset_paths(args.dataset)["models_dir"]

    proj_path = projector_path(models_dir)
    if not os.path.exists(proj_path):
        log.error("Projector não encontrado: %s\n  → rode 02_project_random.py --dataset %s primeiro.",
                  proj_path, args.dataset)
        sys.exit(1)
    with open(proj_path, "rb") as f:
        projector = pickle.load(f)
    n_components = projector.n_components_

    clf_path = classifier_path(models_dir, args.model, n_components)
    if not os.path.exists(clf_path):
        log.error(
            "Classificador não encontrado: %s\n"
            "  → rode primeiro: python src/06_train_final.py --model %s --dataset %s",
            clf_path, args.model, args.dataset,
        )
        sys.exit(1)
    with open(clf_path, "rb") as f:
        clf = pickle.load(f)

    _, vocab_index = build_vocab()
    log.info("Modelo '%s' carregado. Vocabulário: %d k-mers.", args.model, len(vocab_index))

    records = load_fasta_files(INPUT_DIR, args.fasta)
    log.info("Total de sequências a predizer: %d\n", len(records))

    print("=" * 65)
    print(f"  RESULTADOS — modelo: {args.model}  |  dataset: {args.dataset}  |  N={n_components}")
    print("=" * 65)

    for seq_id, seq in records:
        vector = sequence_to_vector(seq, vocab_index, projector)
        if vector is None:
            print(f"  {'[IGNORADA]':<12}  {seq_id}  (sequência muito curta)")
            continue

        X = vector.reshape(1, -1)
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
