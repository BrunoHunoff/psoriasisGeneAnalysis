"""
Módulo 01 — Geração das matrizes posicionais k-mer.

Lê sequências de duas fontes com labels distintos (diretórios em common.DATASETS):
  - tcc1: data/raw/    → positivos (label=1) | data/raw_negativos/    → negativos (label=0)
  - tcc2: data/raw_v2/ → positivos (label=1) | data/raw_negativos_v2/ → negativos (label=0)

Cada sequência individual gera uma matriz one-hot (L-s+1, n_kmers), salva em formato
esparso em {proc_dir}/matrices/{key}_matrix.npz, onde key = {stem}_{idx:03d}. Com o k=3
contínuo do TCC I é (L-2, 8000); --k e --pattern escolhem outra representação, gravada
em {proc_dir}/{representação}/ (ver common.py).

Salva sequence_registry.json com metadados de todas as sequências,
incluindo chave, label, arquivo de origem e shape da matriz.

Uso:
    python src/01_kmer_matrix.py --dataset tcc2
    python src/01_kmer_matrix.py --dataset tcc2 --k 4
    python src/01_kmer_matrix.py --dataset tcc2 --pattern 1101 1011
"""

import os
import sys
import json
import logging
import argparse

from Bio import SeqIO
from scipy import sparse

from common import (FASTA_EXTS, add_dataset_arg, add_representation_args, dataset_paths,
                    feature_names, patterns_from_args, rep_name, sequence_to_kmer_matrix)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

LABEL_POSITIVE = 1
LABEL_NEGATIVE = 0


def load_fasta_dir(directory: str, label: int) -> list[dict]:
    """
    Carrega todas as sequências de todos os FASTAs em directory.

    Retorna lista de dicts com: key, seq_id, protein, label, source_file, seq.
    A chave é {stem}_{idx:03d} onde idx é o índice da sequência dentro do arquivo.
    """
    if not os.path.isdir(directory):
        log.error("Diretório não encontrado: %s", directory)
        sys.exit(1)

    fasta_files = sorted(
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in FASTA_EXTS
    )
    if not fasta_files:
        log.error("Nenhum arquivo .fasta/.fa encontrado em '%s'.", directory)
        sys.exit(1)

    records = []
    for fname in fasta_files:
        stem = os.path.splitext(fname)[0]
        path = os.path.join(directory, fname)
        seqs = list(SeqIO.parse(path, "fasta"))
        log.info("  %s: %d sequência(s)  label=%d", fname, len(seqs), label)
        for idx, rec in enumerate(seqs):
            records.append({
                "key": f"{stem}_{idx:03d}",
                "seq_id": rec.id,
                "protein": stem,
                "label": label,
                "source_file": fname,
                "seq": str(rec.seq).upper(),
            })
    return records


def main():
    parser = argparse.ArgumentParser()
    add_dataset_arg(parser)
    add_representation_args(parser, n_components=False)
    args = parser.parse_args()
    patterns = patterns_from_args(args)
    paths = dataset_paths(args.dataset, patterns)
    raw_pos_dir, raw_neg_dir = paths["pos_dir"], paths["neg_dir"]
    matrices_dir  = os.path.join(paths["proc_dir"], "matrices")
    vocab_path    = os.path.join(paths["proc_dir"], "kmer_vocab.json")
    registry_path = os.path.join(paths["proc_dir"], "sequence_registry.json")
    log.info("Dataset: %s  |  representação: %s %s", args.dataset, rep_name(patterns), list(patterns))

    os.makedirs(matrices_dir, exist_ok=True)

    vocab_list = feature_names(patterns)
    vocab_for_json = {str(idx): kmer for idx, kmer in enumerate(vocab_list)}
    with open(vocab_path, "w") as f:
        json.dump(vocab_for_json, f)
    log.info("Vocabulário salvo: %s (%d k-mers)", vocab_path, len(vocab_list))

    log.info("Carregando positivos de '%s'...", raw_pos_dir)
    pos_records = load_fasta_dir(raw_pos_dir, LABEL_POSITIVE)

    log.info("Carregando negativos de '%s'...", raw_neg_dir)
    neg_records = load_fasta_dir(raw_neg_dir, LABEL_NEGATIVE)

    all_records = pos_records + neg_records
    log.info(
        "Total: %d sequências (%d positivas, %d negativas)",
        len(all_records), len(pos_records), len(neg_records),
    )

    registry = []
    total_windows = 0

    for rec in all_records:
        key = rec["key"]
        seq = rec["seq"]
        try:
            matrix = sequence_to_kmer_matrix(seq, patterns)
        except ValueError as e:
            log.warning("Sequência ignorada (%s): %s", key, e)
            continue

        out_path = os.path.join(matrices_dir, f"{key}_matrix.npz")
        sparse.save_npz(out_path, matrix)
        total_windows += matrix.shape[0]

        registry.append({
            "key": key,
            "seq_id": rec["seq_id"],
            "protein": rec["protein"],
            "label": rec["label"],
            "source_file": rec["source_file"],
            "matrix_shape": list(matrix.shape),
        })
        log.info(
            "%-40s  label=%d  len=%-5d  matrix=%s",
            rec["seq_id"][:40], rec["label"], len(seq), matrix.shape,
        )

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    n_pos = sum(1 for r in registry if r["label"] == LABEL_POSITIVE)
    n_neg = sum(1 for r in registry if r["label"] == LABEL_NEGATIVE)
    log.info("Registry salvo: %s", registry_path)
    log.info("Sequências processadas : %d  (pos=%d, neg=%d)", len(registry), n_pos, n_neg)
    log.info("Total de janelas k-mer : %d", total_windows)


if __name__ == "__main__":
    main()
