"""
Definições compartilhadas pela pipeline: datasets (TCC I / TCC II) e vetorização k-mer.

Cada dataset tem seus próprios diretórios de entrada E de saída, para que rodar o TCC II
nunca sobrescreva os artefatos e relatórios do TCC I (e vice-versa):

  tcc1 — dataset congelado do TCC I (data/raw + data/raw_negativos), saídas em
         data/processed, models/ e reports/ — os mesmos caminhos usados nos quadros do TCC I
  tcc2 — expansão Dand et al. (2025), saídas em data/processed_v2, models/v2 e reports/v2;
         tem conjunto de teste cego (data/teste_cego_v2 + data/teste_cego_negativos_v2)

Os scripts recebem --dataset {tcc1,tcc2} (padrão: DEFAULT_DATASET).
"""

import os
import itertools

import numpy as np
from scipy import sparse

DATASETS = {
    "tcc1": {
        "pos_dir":       "data/raw",
        "neg_dir":       "data/raw_negativos",
        "proc_dir":      "data/processed",
        "models_dir":    "models",
        "reports_dir":   "reports",
        "blind_pos_dir": None,
        "blind_neg_dir": None,
    },
    "tcc2": {
        "pos_dir":       "data/raw_v2",
        "neg_dir":       "data/raw_negativos_v2",
        "proc_dir":      "data/processed_v2",
        "models_dir":    os.path.join("models", "v2"),
        "reports_dir":   os.path.join("reports", "v2"),
        "blind_pos_dir": "data/teste_cego_v2",
        "blind_neg_dir": "data/teste_cego_negativos_v2",
    },
}
DEFAULT_DATASET = "tcc2"

# Proteínas de validação externa do TCC I (data/input/), repetidas no TCC II para comparação
INPUT_DIR = "data/input"

K          = 3
AMINO      = "ACDEFGHIKLMNPQRSTVWY"
FASTA_EXTS = (".fasta", ".fa")


def add_dataset_arg(parser) -> None:
    parser.add_argument(
        "--dataset", choices=list(DATASETS), default=DEFAULT_DATASET,
        help=f"Dataset: tcc1 (congelado do TCC I) ou tcc2 (Dand 2025). Padrão: {DEFAULT_DATASET}.",
    )


def dataset_paths(name: str) -> dict:
    return DATASETS[name]


def projector_path(models_dir: str) -> str:
    return os.path.join(models_dir, "random_projection.pkl")


def classifier_path(models_dir: str, model: str, n_components: int) -> str:
    return os.path.join(models_dir, f"classifier_{model}_n{n_components}.pkl")


def build_vocab(amino: str = AMINO, k: int = K) -> tuple[list[str], dict[str, int]]:
    """Gera todas as combinações de k aminoácidos. Retorna (lista, mapa kmer→índice)."""
    vocab = ["".join(c) for c in itertools.product(amino, repeat=k)]
    return vocab, {kmer: idx for idx, kmer in enumerate(vocab)}


def sequence_to_onehot_matrix(seq: str, vocab_index: dict[str, int], k: int = K) -> sparse.csr_matrix:
    """
    Constrói a matriz posicional one-hot (L-k+1, |vocab|) de uma sequência, em formato
    esparso (CSR): cada linha tem um único 1, na coluna do k-mer daquela posição.
    K-mers com caracteres fora do vocabulário resultam em linha de zeros.

    É a mesma matriz da versão densa, só armazenada sem os zeros: no TCC II a matriz
    global densa teria ~325 mil janelas × 8.000 colunas (~10 GB em float32).
    """
    n_windows = len(seq) - k + 1
    if n_windows <= 0:
        raise ValueError(f"Sequência curta demais para k={k}: len={len(seq)}")
    rows, cols = [], []
    for i in range(n_windows):
        idx = vocab_index.get(seq[i : i + k])
        if idx is not None:
            rows.append(i)
            cols.append(idx)
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n_windows, len(vocab_index)), dtype=np.float32)


def sequence_to_vector(seq: str, vocab_index: dict[str, int], projector) -> np.ndarray | None:
    """
    Vetor (N,) de uma sequência nova: matriz k-mer → projeção ajustada no treino → mean pooling.
    Idêntico ao que 02_project_random.py faz com as sequências de treino.
    """
    try:
        matrix = sequence_to_onehot_matrix(seq, vocab_index)
    except ValueError:
        return None
    return np.asarray(projector.transform(matrix)).mean(axis=0).astype(np.float32)
