"""
Módulo 02 — Random Projection e geração dos vetores finais.

Para cada sequência individual:
  1. Projeta cada janela da matriz k-mer com uma matriz gaussiana (eixos aleatórios, sem
     viés dos dados) — n_kmers → N
  2. Faz mean pooling das janelas projetadas → vetor (N,)

A matriz gaussiana é a do GaussianRandomProjection do scikit-learn (random_state=42) e só
depende do número de colunas e do RANDOM_STATE, então projetar sequência por sequência dá
exatamente os mesmos vetores que ajustar o projetor sobre a matriz global empilhada, como
no TCC I (verificado bit a bit no X do TCC II com N=1000). Ver common.kmer_vectors.

--n-components 0 → sem projeção: o vetor é o mean pooling das linhas one-hot, isto é, a
frequência relativa de cada k-mer (dimensão = número de k-mers, ex. 8000 para k=3).

Saídas ({proc_dir} conforme dataset e representação, ver common.dataset_paths):
  {proc_dir}/X_n{dim}.npy              — (n_seq, dim) uma linha por sequência
  {proc_dir}/y.npy                     — (n_seq,) labels 1/0
  {proc_dir}/metadata.json             — mapeamento índice→{key, seq_id, protein, label}
  {proc_dir}/random_summary_n{dim}.json

Não há projetor salvo: 07/08 refazem a mesma projeção a partir de (representação, N, seed).

Uso:
    python src/02_project_random.py --dataset tcc2 --n-components 1000
    python src/02_project_random.py --dataset tcc2 --k 4 --n-components 4000
    python src/02_project_random.py --dataset tcc2 --pattern 1101 --n-components 0
"""

import os
import sys
import json
import logging
import argparse
import numpy as np

from scipy import sparse

from common import (RANDOM_STATE, add_dataset_arg, add_representation_args, dataset_paths,
                    kmer_vectors, n_kmer_features, patterns_from_args, rep_name, vector_dim,
                    x_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def load_registry(path: str) -> list[dict]:
    """Carrega o registry de sequências gerado por 01_kmer_matrix.py."""
    if not os.path.exists(path):
        log.error("%s não encontrado — rode 01_kmer_matrix.py com a mesma representação primeiro.", path)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def load_matrices(registry: list[dict], matrices_dir: str) -> list[sparse.csr_matrix]:
    """
    Carrega as matrizes individuais na ordem do registry.

    A ordem determina as linhas de X — deve ser idêntica à de y e do metadata.
    """
    matrices = []
    for entry in registry:
        path = os.path.join(matrices_dir, f"{entry['key']}_matrix.npz")
        if not os.path.exists(path):
            log.error("Matriz não encontrada: %s — rode 01_kmer_matrix.py com o mesmo --dataset "
                      "e a mesma representação.", path)
            sys.exit(1)
        matrices.append(sparse.load_npz(path))
    return matrices


def parse_args() -> argparse.Namespace:
    """Parseia argumentos de linha de comando."""
    parser = argparse.ArgumentParser()
    add_dataset_arg(parser)
    add_representation_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    patterns = patterns_from_args(args)
    n_components = args.n_components
    n_features = n_kmer_features(patterns)
    dim = vector_dim(patterns, n_components)
    proc_dir = dataset_paths(args.dataset, patterns)["proc_dir"]
    matrices_dir  = os.path.join(proc_dir, "matrices")
    registry_path = os.path.join(proc_dir, "sequence_registry.json")
    summary_path  = os.path.join(proc_dir, f"random_summary_n{dim}.json")
    out_x_path    = x_path(proc_dir, dim)
    y_path        = os.path.join(proc_dir, "y.npy")
    metadata_path = os.path.join(proc_dir, "metadata.json")
    log.info("Dataset: %s  |  representação: %s %s", args.dataset, rep_name(patterns), list(patterns))

    if n_components < 0 or n_components >= n_features:
        log.error("--n-components %d inválido para %d k-mers: use 0 < N < %d, ou 0 para sem projeção.",
                  n_components, n_features, n_features)
        sys.exit(1)

    registry = load_registry(registry_path)
    n_seq = len(registry)
    log.info("Registry carregado: %d sequências", n_seq)

    matrices = load_matrices(registry, matrices_dir)
    if matrices[0].shape[1] != n_features:
        log.error("Matrizes com %d colunas, esperado %d para %s — rode 01_kmer_matrix.py de novo.",
                  matrices[0].shape[1], n_features, rep_name(patterns))
        sys.exit(1)
    total_windows = sum(m.shape[0] for m in matrices)
    log.info("Matrizes carregadas: %d janelas × %d k-mers (%d valores não nulos)",
             total_windows, n_features, sum(m.nnz for m in matrices))

    if n_components:
        log.info("Projeção gaussiana: %d → %d componentes + mean pooling", n_features, n_components)
    else:
        log.info("Sem projeção: mean pooling direto (frequência dos %d k-mers)", n_features)
    X = kmer_vectors(matrices, n_components)
    log.info("Vetores concluídos: %s", X.shape)

    y = np.array([entry["label"] for entry in registry], dtype=np.int8)
    metadata = {
        str(i): {
            "key": entry["key"],
            "seq_id": entry["seq_id"],
            "protein": entry["protein"],
            "label": entry["label"],
            "source_file": entry["source_file"],
        }
        for i, entry in enumerate(registry)
    }

    np.save(out_x_path, X)
    np.save(y_path, y)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    with open(summary_path, "w") as f:
        json.dump({
            "method": "random_projection" if n_components else "sem_projecao",
            "dataset": args.dataset,
            "representation": rep_name(patterns),
            "patterns": list(patterns),
            "n_kmer_features": n_features,
            "n_components": dim,
            "projection": bool(n_components),
            "random_state": RANDOM_STATE,
            "n_sequences": n_seq,
            "n_positive": int(y.sum()),
            "n_negative": int((y == 0).sum()),
            "X_shape": list(X.shape),
            "total_windows": total_windows,
        }, f, indent=2)

    log.info("X salvo: %s  %s", out_x_path, X.shape)
    log.info("y salvo: %s  (pos=%d, neg=%d)", y_path, int(y.sum()), int((y == 0).sum()))
    log.info("metadata salvo: %s", metadata_path)


if __name__ == "__main__":
    main()
