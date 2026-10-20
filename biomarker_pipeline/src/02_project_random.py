"""
Módulo 02 — Random Projection e geração dos vetores finais.

Para cada sequência individual:
  1. Empilha todas as matrizes k-mer em matriz global (esparsa)
  2. Aplica GaussianRandomProjection (eixos aleatórios, sem viés dos dados)
  3. Reagrupa linhas por sequência e aplica mean pooling → vetor (N,)

A projeção gaussiana depende só do número de colunas (8.000) e do RANDOM_STATE, então a
matriz esparsa produz exatamente os mesmos vetores que a versão densa usada no TCC I.

Saídas principais para classificação ({proc_dir} conforme o dataset, ver common.DATASETS):
  {proc_dir}/X.npy          — (n_seq, N_COMPONENTS) uma linha por sequência
  {proc_dir}/y.npy          — (n_seq,) labels 1/0
  {proc_dir}/metadata.json  — mapeamento índice→{key, seq_id, protein, label}
  {models_dir}/random_projection.pkl

Uso:
    python src/02_project_random.py --dataset tcc2 --n-components 1000
"""

import os
import sys
import json
import pickle
import logging
import argparse
import numpy as np

from scipy import sparse
from sklearn.random_projection import GaussianRandomProjection

from common import add_dataset_arg, dataset_paths, projector_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

N_COMPONENTS = 1000   # configuração final do TCC I
RANDOM_STATE = 42


def load_registry(path: str) -> list[dict]:
    """Carrega o registry de sequências gerado por 01_kmer_matrix.py."""
    if not os.path.exists(path):
        log.error("%s não encontrado — rode 01_kmer_matrix.py primeiro.", path)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def load_global_matrix(registry: list[dict], matrices_dir: str) -> sparse.csr_matrix:
    """
    Carrega as matrizes individuais na ordem do registry e empilha em matriz global.

    A ordem determina o reagrupamento posterior — deve ser idêntica à usada no pooling.
    """
    blocks = []
    for entry in registry:
        path = os.path.join(matrices_dir, f"{entry['key']}_matrix.npz")
        if not os.path.exists(path):
            log.error("Matriz não encontrada: %s — rode 01_kmer_matrix.py com o mesmo --dataset.", path)
            sys.exit(1)
        blocks.append(sparse.load_npz(path))
    return sparse.vstack(blocks, format="csr")


def parse_args() -> argparse.Namespace:
    """Parseia argumentos de linha de comando."""
    parser = argparse.ArgumentParser()
    add_dataset_arg(parser)
    parser.add_argument(
        "--n-components",
        type=int,
        default=N_COMPONENTS,
        help=f"Número de componentes da projeção (padrão: {N_COMPONENTS}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    n_components = args.n_components
    paths = dataset_paths(args.dataset)
    proc_dir, models_dir = paths["proc_dir"], paths["models_dir"]
    matrices_dir  = os.path.join(proc_dir, "matrices")
    vectors_dir   = os.path.join(proc_dir, "vectors")
    registry_path = os.path.join(proc_dir, "sequence_registry.json")
    model_path    = projector_path(models_dir)
    summary_path  = os.path.join(proc_dir, "random_summary.json")
    x_path        = os.path.join(proc_dir, "X.npy")
    y_path        = os.path.join(proc_dir, "y.npy")
    metadata_path = os.path.join(proc_dir, "metadata.json")
    log.info("Dataset: %s", args.dataset)

    for d in (vectors_dir, models_dir):
        os.makedirs(d, exist_ok=True)

    registry = load_registry(registry_path)
    n_seq = len(registry)
    log.info("Registry carregado: %d sequências", n_seq)

    X_global = load_global_matrix(registry, matrices_dir)
    total_windows, n_features = X_global.shape
    log.info("Matriz global empilhada: %s  (esparsa, %d valores não nulos)",
             X_global.shape, X_global.nnz)

    log.info("N componentes: %d", n_components)
    transformer = GaussianRandomProjection(n_components=n_components, random_state=RANDOM_STATE)
    X_projected = np.asarray(transformer.fit_transform(X_global))
    log.info("Projeção concluída: %s", X_projected.shape)

    with open(model_path, "wb") as f:
        pickle.dump(transformer, f)
    log.info("Modelo salvo: %s", model_path)

    # Mean pooling por sequência
    X = np.zeros((n_seq, n_components), dtype=np.float32)
    y = np.zeros(n_seq, dtype=np.int8)
    metadata = {}

    start = 0
    for i, entry in enumerate(registry):
        n_rows = entry["matrix_shape"][0]
        rows = X_projected[start : start + n_rows]
        vector = rows.mean(axis=0).astype(np.float32)

        X[i] = vector
        y[i] = entry["label"]
        metadata[str(i)] = {
            "key": entry["key"],
            "seq_id": entry["seq_id"],
            "protein": entry["protein"],
            "label": entry["label"],
            "source_file": entry["source_file"],
        }

        vec_path = os.path.join(vectors_dir, f"{entry['key']}_vector.npy")
        np.save(vec_path, vector)
        start += n_rows

        log.info(
            "[%3d] %-40s  label=%d  rows=%-5d  vector=%s",
            i, entry["seq_id"][:40], entry["label"], n_rows, vector.shape,
        )

    np.save(x_path, X)
    np.save(y_path, y)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    with open(summary_path, "w") as f:
        json.dump({
            "method": "random_projection",
            "dataset": args.dataset,
            "n_components": n_components,
            "random_state": RANDOM_STATE,
            "n_sequences": n_seq,
            "n_positive": int(y.sum()),
            "n_negative": int((y == 0).sum()),
            "X_shape": list(X.shape),
            "total_windows": total_windows,
        }, f, indent=2)

    log.info("X salvo: %s  %s", x_path, X.shape)
    log.info("y salvo: %s  (pos=%d, neg=%d)", y_path, int(y.sum()), int((y == 0).sum()))
    log.info("metadata salvo: %s", metadata_path)


if __name__ == "__main__":
    main()
