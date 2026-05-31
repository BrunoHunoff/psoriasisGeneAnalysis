"""
Módulo 03 — Projeção Random (GaussianRandomProjection).

Usa o mesmo N escolhido pelo SVD (lido de data/processed/svd/summary.json)
para garantir comparação justa. Aplica projeção aleatória sobre a mesma
matriz global e salva vetores por proteína via mean pooling.

Diferença-chave em relação ao SVD: os eixos de projeção são gerados
aleatoriamente, sem nenhum viés introduzido pelos dados de treino.
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

PROC_DIR            = "data/processed"
MATRICES_DIR        = os.path.join(PROC_DIR, "matrices")
RANDOM_DIR          = os.path.join(PROC_DIR, "random")
RANDOM_VECTORS_DIR  = os.path.join(RANDOM_DIR, "vectors")
MODELS_DIR          = "models"
SHAPES_PATH         = os.path.join(PROC_DIR, "matrix_shapes.json")
IDS_PATH            = os.path.join(PROC_DIR, "sequence_ids.json")
SVD_SUMMARY_PATH    = os.path.join(PROC_DIR, "svd", "summary.json")
RANDOM_MODEL_PATH   = os.path.join(MODELS_DIR, "random_projection.pkl")
RANDOM_SUMMARY_PATH = os.path.join(RANDOM_DIR, "summary.json")

N_COMPONENTS = 500
RANDOM_STATE = 42


def load_global_matrix(matrices_dir: str, shapes: dict) -> np.ndarray:
    """
    Carrega matrizes individuais na ordem de shapes e empilha em matriz global.

    A ordem deve ser idêntica à usada em 02_project_svd.py para que os
    reagrupamentos por proteína sejam comparáveis.
    """
    blocks = []
    for stem in shapes:
        path = os.path.join(matrices_dir, f"{stem}_matrix.npy")
        if not os.path.exists(path):
            log.error("Matriz não encontrada: %s — rode 01_kmer_matrix.py primeiro.", path)
            sys.exit(1)
        blocks.append(np.load(path))
    return np.vstack(blocks)


def resolve_n_components(svd_summary_path: str, fallback: int) -> int:
    """
    Lê o N escolhido pelo SVD para usar o mesmo valor na projeção aleatória.
    Se o sumário não existir, usa o fallback.
    """
    if os.path.exists(svd_summary_path):
        with open(svd_summary_path) as f:
            svd_summary = json.load(f)
        n = svd_summary.get("n_components", fallback)
        log.info("N lido do sumário SVD: %d", n)
        return n
    log.warning(
        "Sumário SVD não encontrado em '%s' — usando N_COMPONENTS=%d como fallback.",
        svd_summary_path, fallback,
    )
    return fallback


def main():
    for d in (RANDOM_VECTORS_DIR, MODELS_DIR):
        os.makedirs(d, exist_ok=True)

    for path, hint in ((SHAPES_PATH, "01_kmer_matrix.py"), (IDS_PATH, "01_kmer_matrix.py")):
        if not os.path.exists(path):
            log.error("%s não encontrado — rode %s primeiro.", path, hint)
            sys.exit(1)

    if not os.path.exists(SVD_SUMMARY_PATH):
        log.error(
            "Sumário SVD não encontrado: %s — rode 02_project_svd.py primeiro.",
            SVD_SUMMARY_PATH,
        )
        sys.exit(1)

    with open(SHAPES_PATH) as f:
        shapes = json.load(f)
    with open(IDS_PATH) as f:
        seq_entries = json.load(f)

    n_components = resolve_n_components(SVD_SUMMARY_PATH, N_COMPONENTS)

    X_global = load_global_matrix(MATRICES_DIR, shapes)
    total_rows, n_features = X_global.shape
    log.info("Matriz global empilhada: %s  (%.1f MB)",
             X_global.shape, X_global.nbytes / 1e6)

    transformer = GaussianRandomProjection(n_components=n_components, random_state=RANDOM_STATE)
    X_reduced = transformer.fit_transform(X_global)
    log.info("X_reduced shape (Random Projection): %s", X_reduced.shape)

    with open(RANDOM_MODEL_PATH, "wb") as f:
        pickle.dump(transformer, f)
    log.info("Modelo Random Projection salvo: %s", RANDOM_MODEL_PATH)

    protein_vectors = {}
    start = 0
    for stem, (n_rows, _) in shapes.items():
        protein_rows = X_reduced[start : start + n_rows]
        vector = protein_rows.mean(axis=0)
        protein_vectors[stem] = vector
        vec_path = os.path.join(RANDOM_VECTORS_DIR, f"{stem}_vector.npy")
        np.save(vec_path, vector)
        start += n_rows

        seq_id = next(
            (e["sequence_id"] for e in seq_entries if e["file_stem"] == stem), stem
        )
        log.info(
            "%-30s  rows=%d  pooling: %s→%s  salvo: %s",
            seq_id[:30], n_rows, protein_rows.shape, vector.shape, vec_path,
        )

    all_vectors = np.stack(list(protein_vectors.values()))
    profile_mean = all_vectors.mean(axis=0)
    profile_std = all_vectors.std(axis=0)
    np.save(os.path.join(RANDOM_DIR, "profile_mean.npy"), profile_mean)
    np.save(os.path.join(RANDOM_DIR, "profile_std.npy"), profile_std)
    log.info("Perfil médio e desvio padrão salvos em %s", RANDOM_DIR)

    summary = {
        "method": "random_projection",
        "n_components": n_components,
        "vector_shape": [n_components],
        "proteins": {
            stem: {
                "n_rows": shapes[stem][0],
                "vector_path": os.path.join(RANDOM_VECTORS_DIR, f"{stem}_vector.npy"),
            }
            for stem in shapes
        },
    }
    with open(RANDOM_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Sumário Random Projection salvo: %s", RANDOM_SUMMARY_PATH)

    log.info("N componentes            : %d", n_components)
    log.info("Shape do vetor por prot. : (%d,)", n_components)


if __name__ == "__main__":
    main()
