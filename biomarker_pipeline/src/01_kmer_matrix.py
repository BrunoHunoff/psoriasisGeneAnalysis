"""
Módulo 01 — Geração das matrizes posicionais k-mer.

Para cada proteína, desliza uma janela de tamanho k=3 sobre a sequência e
constrói uma matriz one-hot: cada linha i contém 1 na coluna do 3-mer
encontrado na posição i e 0 nas demais.

Shape da matriz por proteína: (L - k + 1, |vocab|) = (L-2, 8000)

Essa é a representação "3D" descrita pelo orientador: rica em informação
posicional, mas ainda incomparável entre proteínas de tamanhos diferentes.
O SVD (módulo 02) fará a compressão para um vetor fixo comparável.
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

RAW_DIR      = "data/raw"
PROC_DIR     = "data/processed"
MATRICES_DIR = os.path.join(PROC_DIR, "matrices")
K            = 3
AMINO        = "ACDEFGHIKLMNPQRSTVWY"
FASTA_EXTS   = (".fasta", ".fa")
VOCAB_PATH   = os.path.join(PROC_DIR, "kmer_vocab.json")
IDS_PATH     = os.path.join(PROC_DIR, "sequence_ids.json")
SHAPES_PATH  = os.path.join(PROC_DIR, "matrix_shapes.json")


def build_vocab(amino: str, k: int) -> tuple[list[str], dict[str, int]]:
    """
    Gera todas as combinações possíveis de k aminoácidos.
    Retorna (lista ordenada, mapa kmer→índice).
    """
    vocab = ["".join(c) for c in itertools.product(amino, repeat=k)]
    return vocab, {kmer: idx for idx, kmer in enumerate(vocab)}


def sequence_to_onehot_matrix(seq: str, vocab_index: dict[str, int], k: int) -> np.ndarray:
    """
    Constrói a matriz posicional one-hot para uma sequência.

    Para cada posição i em [0, L-k], extrai o k-mer seq[i:i+k] e coloca 1
    na coluna correspondente ao índice do k-mer no vocabulário.

    Retorna matriz de shape (L - k + 1, |vocab|), dtype float32.
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


def load_fasta_files(raw_dir: str) -> list[tuple[str, str, str]]:
    """
    Carrega todos os FASTAs de raw_dir.
    Retorna lista de (file_stem, sequence_id, sequence).
    """
    if not os.path.isdir(raw_dir):
        log.error("Diretório não encontrado: %s", raw_dir)
        sys.exit(1)

    fasta_files = sorted(
        f for f in os.listdir(raw_dir)
        if os.path.splitext(f)[1].lower() in FASTA_EXTS
    )
    if not fasta_files:
        log.error(
            "Nenhum arquivo .fasta/.fa encontrado em '%s'. "
            "Adicione os FASTAs dos biomarcadores antes de rodar o pipeline.",
            raw_dir,
        )
        sys.exit(1)

    records = []
    for fname in fasta_files:
        stem = os.path.splitext(fname)[0]
        path = os.path.join(raw_dir, fname)
        seqs = list(SeqIO.parse(path, "fasta"))
        log.info("Arquivo '%s': %d sequência(s).", fname, len(seqs))
        for rec in seqs:
            records.append((stem, rec.id, str(rec.seq).upper()))
    return records


def main():
    os.makedirs(MATRICES_DIR, exist_ok=True)

    records = load_fasta_files(RAW_DIR)
    vocab_list, vocab_index = build_vocab(AMINO, K)

    vocab_for_json = {str(idx): kmer for idx, kmer in enumerate(vocab_list)}
    with open(VOCAB_PATH, "w") as f:
        json.dump(vocab_for_json, f)
    log.info("Vocabulário salvo: %s (%d k-mers)", VOCAB_PATH, len(vocab_list))

    sequence_ids = []
    matrix_shapes = {}   # {stem: [n_rows, n_cols]} — necessário para reagrupar após SVD global
    total_rows = 0

    for stem, seq_id, seq in records:
        matrix = sequence_to_onehot_matrix(seq, vocab_index, K)
        out_path = os.path.join(MATRICES_DIR, f"{stem}_matrix.npy")
        np.save(out_path, matrix)
        sequence_ids.append({"file_stem": stem, "sequence_id": seq_id})
        matrix_shapes[stem] = list(matrix.shape)
        total_rows += matrix.shape[0]
        log.info(
            "%-30s  len=%-4d  matrix=%s  linhas_acumuladas=%d",
            seq_id, len(seq), matrix.shape, total_rows,
        )

    with open(IDS_PATH, "w") as f:
        json.dump(sequence_ids, f, indent=2)
    with open(SHAPES_PATH, "w") as f:
        json.dump(matrix_shapes, f, indent=2)
    log.info("IDs salvos: %s", IDS_PATH)
    log.info("Shapes salvos: %s  (total de linhas: %d)", SHAPES_PATH, total_rows)


if __name__ == "__main__":
    main()
