"""
Módulo 02 — Random Projection e geração dos vetores finais.

Para cada sequência individual:
  1. Empilha todas as matrizes k-mer em matriz global
  2. Aplica GaussianRandomProjection (eixos aleatórios, sem viés dos dados)
  3. Reagrupa linhas por sequência e aplica mean pooling → vetor (N,)

Saídas principais para classificação:
  data/processed/X.npy          — (n_seq, N_COMPONENTS) uma linha por sequência
  data/processed/y.npy          — (n_seq,) labels 1/0
  data/processed/metadata.json  — mapeamento índice→{key, seq_id, protein, label}
"""

import os
import sys
import json
import pickle
import logging
import numpy as np

from sklearn.random_projection import GaussianRandomProjection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PROC_DIR        = "data/processed"
MATRICES_DIR    = os.path.join(PROC_DIR, "matrices")
VECTORS_DIR     = os.path.join(PROC_DIR, "vectors")
MODELS_DIR      = "models"
REGISTRY_PATH   = os.path.join(PROC_DIR, "sequence_registry.json")
MODEL_PATH      = os.path.join(MODELS_DIR, "random_projection.pkl")
SUMMARY_PATH    = os.path.join(PROC_DIR, "random_summary.json")
X_PATH          = os.path.join(PROC_DIR, "X.npy")
Y_PATH          = os.path.join(PROC_DIR, "y.npy")
METADATA_PATH   = os.path.join(PROC_DIR, "metadata.json")

N_COMPONENTS = 500
RANDOM_STATE = 42


def load_registry(path: str) -> list[dict]:
    """Carrega o registry de sequências gerado por 01_kmer_matrix.py."""
    if not os.path.exists(path):
        log.error("%s não encontrado — rode 01_kmer_matrix.py primeiro.", path)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def load_global_matrix(registry: list[dict], matrices_dir: str) -> np.ndarray:
    """
    Carrega as matrizes individuais na ordem do registry e empilha em matriz global.

    A ordem determina o reagrupamento posterior — deve ser idêntica à usada no pooling.
    """
    blocks = []
    for entry in registry:
        path = os.path.join(matrices_dir, f"{entry['key']}_matrix.npy")
        if not os.path.exists(path):
            log.error("Matriz não encontrada: %s", path)
            sys.exit(1)
        blocks.append(np.load(path))
    return np.vstack(blocks)


def main():
    for d in (VECTORS_DIR, MODELS_DIR):
        os.makedirs(d, exist_ok=True)

    registry = load_registry(REGISTRY_PATH)
    n_seq = len(registry)
    log.info("Registry carregado: %d sequências", n_seq)

    X_global = load_global_matrix(registry, MATRICES_DIR)
    total_windows, n_features = X_global.shape
    log.info("Matriz global empilhada: %s  (%.1f MB)",
             X_global.shape, X_global.nbytes / 1e6)

    transformer = GaussianRandomProjection(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    X_projected = transformer.fit_transform(X_global)   # (total_windows, N_COMPONENTS)
    log.info("Projeção concluída: %s", X_projected.shape)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(transformer, f)
    log.info("Modelo salvo: %s", MODEL_PATH)

    # Mean pooling por sequência
    X = np.zeros((n_seq, N_COMPONENTS), dtype=np.float32)
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

        vec_path = os.path.join(VECTORS_DIR, f"{entry['key']}_vector.npy")
        np.save(vec_path, vector)
        start += n_rows

        log.info(
            "[%3d] %-40s  label=%d  rows=%-5d  vector=%s",
            i, entry["seq_id"][:40], entry["label"], n_rows, vector.shape,
        )

    np.save(X_PATH, X)
    np.save(Y_PATH, y)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    with open(SUMMARY_PATH, "w") as f:
        json.dump({
            "method": "random_projection",
            "n_components": N_COMPONENTS,
            "random_state": RANDOM_STATE,
            "n_sequences": n_seq,
            "n_positive": int(y.sum()),
            "n_negative": int((y == 0).sum()),
            "X_shape": list(X.shape),
            "total_windows": total_windows,
        }, f, indent=2)

    log.info("X salvo: %s  %s", X_PATH, X.shape)
    log.info("y salvo: %s  (pos=%d, neg=%d)", Y_PATH, int(y.sum()), int((y == 0).sum()))
    log.info("metadata salvo: %s", METADATA_PATH)


if __name__ == "__main__":
    main()
