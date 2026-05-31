"""
Módulo 02 — Projeção SVD global.

Empilha todas as matrizes posicionais em uma única matriz global, aplica
TruncatedSVD com o menor N que capture >= VARIANCE_THRESHOLD de variância
(ou N_COMPONENTS se o threshold não for atingido), e salva os vetores
por proteína via mean pooling.
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import TruncatedSVD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PROC_DIR          = "data/processed"
MATRICES_DIR      = os.path.join(PROC_DIR, "matrices")
SVD_DIR           = os.path.join(PROC_DIR, "svd")
SVD_VECTORS_DIR   = os.path.join(SVD_DIR, "vectors")
REPORTS_DIR       = "reports"
MODELS_DIR        = "models"
SHAPES_PATH       = os.path.join(PROC_DIR, "matrix_shapes.json")
IDS_PATH          = os.path.join(PROC_DIR, "sequence_ids.json")
SVD_MODEL_PATH    = os.path.join(MODELS_DIR, "svd_global.pkl")
SVD_SUMMARY_PATH  = os.path.join(SVD_DIR, "summary.json")
SCREE_PLOT_PATH   = os.path.join(REPORTS_DIR, "scree_plot_svd.png")

N_COMPONENTS       = 500
N_CANDIDATES       = [50, 100, 200, 300, N_COMPONENTS]
VARIANCE_THRESHOLD = 0.90
RANDOM_STATE       = 42


def load_global_matrix(matrices_dir: str, shapes: dict) -> np.ndarray:
    """
    Carrega matrizes individuais na ordem de shapes e empilha em matriz global.

    A ordem deve ser a mesma usada no reagrupamento posterior.
    """
    blocks = []
    for stem in shapes:
        path = os.path.join(matrices_dir, f"{stem}_matrix.npy")
        if not os.path.exists(path):
            log.error("Matriz não encontrada: %s — rode 01_kmer_matrix.py primeiro.", path)
            sys.exit(1)
        blocks.append(np.load(path))
    return np.vstack(blocks)


def plot_scree(
    candidates: list,
    variances: list,
    chosen: int,
    threshold: float,
    out_path: str,
) -> None:
    """Gera e salva o scree plot com marcador no N escolhido."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(candidates, variances, marker="o", linewidth=2, color="#2c7bb6",
            label="Variância acumulada")
    ax.axhline(threshold, linestyle="--", color="#d7191c", linewidth=1.5,
               label=f"Threshold {threshold:.0%}")
    chosen_var = variances[candidates.index(chosen)]
    ax.scatter([chosen], [chosen_var], s=140, zorder=5, color="#1a9641",
               label=f"Escolhido N={chosen} ({chosen_var:.2%})")
    ax.set_xlabel("Número de componentes N", fontsize=12)
    ax.set_ylabel("Variância explicada acumulada", fontsize=12)
    ax.set_title("Scree plot — SVD global dos biomarcadores da psoríase", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Scree plot salvo: %s", out_path)


def select_n_components(candidates: list, variances: list, threshold: float) -> int:
    """Retorna o menor N que atinge o threshold; se nenhum atingir, retorna o maior candidato."""
    for n, var in zip(candidates, variances):
        if var >= threshold:
            return n
    return candidates[-1]


def main():
    for d in (SVD_VECTORS_DIR, REPORTS_DIR, MODELS_DIR):
        os.makedirs(d, exist_ok=True)

    for path, hint in ((SHAPES_PATH, "01_kmer_matrix.py"), (IDS_PATH, "01_kmer_matrix.py")):
        if not os.path.exists(path):
            log.error("%s não encontrado — rode %s primeiro.", path, hint)
            sys.exit(1)

    with open(SHAPES_PATH) as f:
        shapes = json.load(f)
    with open(IDS_PATH) as f:
        seq_entries = json.load(f)

    X_global = load_global_matrix(MATRICES_DIR, shapes)
    total_rows, n_features = X_global.shape
    log.info("Matriz global empilhada: %s  (%.1f MB)",
             X_global.shape, X_global.nbytes / 1e6)

    max_possible = min(total_rows, n_features) - 1
    candidates = sorted(set(n for n in N_CANDIDATES if n <= max_possible))
    if not candidates:
        candidates = [max_possible]

    log.info("Testando N candidatos: %s", candidates)
    variances = []
    for n in candidates:
        svd_tmp = TruncatedSVD(n_components=n, random_state=RANDOM_STATE)
        svd_tmp.fit(X_global)
        cum_var = float(svd_tmp.explained_variance_ratio_.sum())
        variances.append(cum_var)
        log.info("  N=%-3d → variância acumulada: %.4f", n, cum_var)

    chosen_n = select_n_components(candidates, variances, VARIANCE_THRESHOLD)
    chosen_var = variances[candidates.index(chosen_n)]
    log.info("N escolhido: %d  (variância=%.4f)", chosen_n, chosen_var)

    plot_scree(candidates, variances, chosen_n, VARIANCE_THRESHOLD, SCREE_PLOT_PATH)

    svd = TruncatedSVD(n_components=chosen_n, random_state=RANDOM_STATE)
    X_reduced = svd.fit_transform(X_global)
    log.info("X_reduced shape: %s", X_reduced.shape)

    with open(SVD_MODEL_PATH, "wb") as f:
        pickle.dump(svd, f)
    log.info("Modelo SVD salvo: %s", SVD_MODEL_PATH)

    protein_vectors = {}
    start = 0
    for stem, (n_rows, _) in shapes.items():
        protein_rows = X_reduced[start : start + n_rows]
        vector = protein_rows.mean(axis=0)
        protein_vectors[stem] = vector
        vec_path = os.path.join(SVD_VECTORS_DIR, f"{stem}_vector.npy")
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
    np.save(os.path.join(SVD_DIR, "profile_mean.npy"), profile_mean)
    np.save(os.path.join(SVD_DIR, "profile_std.npy"), profile_std)
    log.info("Perfil médio e desvio padrão salvos em %s", SVD_DIR)

    summary = {
        "method": "svd",
        "n_components": chosen_n,
        "variance_captured": round(chosen_var, 6),
        "variance_threshold": VARIANCE_THRESHOLD,
        "candidates_tested": candidates,
        "variance_per_candidate": {str(n): round(v, 6) for n, v in zip(candidates, variances)},
        "vector_shape": [chosen_n],
        "proteins": {
            stem: {
                "n_rows": shapes[stem][0],
                "vector_path": os.path.join(SVD_VECTORS_DIR, f"{stem}_vector.npy"),
            }
            for stem in shapes
        },
    }
    with open(SVD_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Sumário SVD salvo: %s", SVD_SUMMARY_PATH)

    log.info("N escolhido              : %d", chosen_n)
    log.info("Variância capturada      : %.4f", chosen_var)
    log.info("Shape do vetor por prot. : (%d,)", chosen_n)


if __name__ == "__main__":
    main()
