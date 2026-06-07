"""
Módulo 01 — Geração das matrizes posicionais k-mer.

Lê sequências de duas fontes com labels distintos:
  - data/raw/          → positivos (label=1): biomarcadores da psoríase
  - data/raw_negativos/ → negativos (label=0): housekeeping / não ligados

Cada sequência individual gera uma matriz one-hot (L-2, 8000) salva em
data/processed/matrices/{key}_matrix.npy, onde key = {stem}_{idx:03d}.

Salva sequence_registry.json com metadados de todas as sequências,
incluindo chave, label, arquivo de origem e shape da matriz.
"""

import os
import sys
import json
import logging
import itertools
import numpy as np

from Bio import SeqIO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

RAW_POS_DIR   = "data/raw"
RAW_NEG_DIR   = "data/raw_negativos"
PROC_DIR      = "data/processed"
MATRICES_DIR  = os.path.join(PROC_DIR, "matrices")
VOCAB_PATH    = os.path.join(PROC_DIR, "kmer_vocab.json")
REGISTRY_PATH = os.path.join(PROC_DIR, "sequence_registry.json")

K          = 3
AMINO      = "ACDEFGHIKLMNPQRSTVWY"
FASTA_EXTS = (".fasta", ".fa")

LABEL_POSITIVE = 1
LABEL_NEGATIVE = 0


def build_vocab(amino: str, k: int) -> tuple[list[str], dict[str, int]]:
    """Gera todas as combinações de k aminoácidos. Retorna (lista, mapa kmer→índice)."""
    vocab = ["".join(c) for c in itertools.product(amino, repeat=k)]
    return vocab, {kmer: idx for idx, kmer in enumerate(vocab)}


def sequence_to_onehot_matrix(seq: str, vocab_index: dict[str, int], k: int) -> np.ndarray:
    """
    Constrói matriz posicional one-hot (L-k+1, |vocab|) para uma sequência.

    Cada linha i = 1 na coluna do k-mer na posição i, 0 nas demais.
    K-mers com caracteres fora do vocabulário resultam em linha de zeros.
    """
    n_vocab = len(vocab_index)
    n_windows = len(seq) - k + 1
    if n_windows <= 0:
        raise ValueError(f"Sequência curta demais para k={k}: len={len(seq)}")

    matrix = np.zeros((n_windows, n_vocab), dtype=np.float32)
    for i in range(n_windows):
        kmer = seq[i : i + k]
        if kmer in vocab_index:
            matrix[i, vocab_index[kmer]] = 1.0
    return matrix


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
    os.makedirs(MATRICES_DIR, exist_ok=True)

    vocab_list, vocab_index = build_vocab(AMINO, K)
    vocab_for_json = {str(idx): kmer for idx, kmer in enumerate(vocab_list)}
    with open(VOCAB_PATH, "w") as f:
        json.dump(vocab_for_json, f)
    log.info("Vocabulário salvo: %s (%d k-mers)", VOCAB_PATH, len(vocab_list))

    log.info("Carregando positivos de '%s'...", RAW_POS_DIR)
    pos_records = load_fasta_dir(RAW_POS_DIR, LABEL_POSITIVE)

    log.info("Carregando negativos de '%s'...", RAW_NEG_DIR)
    neg_records = load_fasta_dir(RAW_NEG_DIR, LABEL_NEGATIVE)

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
            matrix = sequence_to_onehot_matrix(seq, vocab_index, K)
        except ValueError as e:
            log.warning("Sequência ignorada (%s): %s", key, e)
            continue

        out_path = os.path.join(MATRICES_DIR, f"{key}_matrix.npy")
        np.save(out_path, matrix)
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

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    n_pos = sum(1 for r in registry if r["label"] == LABEL_POSITIVE)
    n_neg = sum(1 for r in registry if r["label"] == LABEL_NEGATIVE)
    log.info("Registry salvo: %s", REGISTRY_PATH)
    log.info("Sequências processadas : %d  (pos=%d, neg=%d)", len(registry), n_pos, n_neg)
    log.info("Total de janelas k-mer : %d", total_windows)


if __name__ == "__main__":
    main()
