"""
Módulo 02 — SVD global e geração dos vetores de representação.

Empilha todas as matrizes posicionais em uma única matriz global (3498, 8000),
aplica TruncatedSVD uma única vez e reagrupa as linhas por proteína para
calcular o vetor final via mean pooling.

Por que SVD global e não por proteína:
- O SVD aprende um espaço latente compartilhado entre todas as proteínas.
- A matriz global tem 3498 linhas, permitindo N >> 186 (máximo que o SVD
  por proteína alcançava pela menor sequência), com variância alta e uniforme.
- O reagrupamento garante que cada proteína contribui com suas próprias linhas
  e recebe de volta exatamente a sua fatia do X_reduced.
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

PROC_DIR         = "data/processed"
MATRICES_DIR     = os.path.join(PROC_DIR, "matrices")
VECTORS_DIR      = os.path.join(PROC_DIR, "vectors")
REPORTS_DIR      = "reports"
MODELS_DIR       = "models"
SHAPES_PATH      = os.path.join(PROC_DIR, "matrix_shapes.json")
IDS_PATH         = os.path.join(PROC_DIR, "sequence_ids.json")
SVD_MODEL_PATH   = os.path.join(MODELS_DIR, "svd_global.pkl")
SVD_SUMMARY_PATH = os.path.join(PROC_DIR, "svd_summary.json")
SCREE_PLOT_PATH  = os.path.join(REPORTS_DIR, "scree_plot_global.png")

N_CANDIDATES       = [50, 100, 200, 300, 500]
VARIANCE_THRESHOLD = 0.90
RANDOM_STATE       = 42


def load_global_matrix(matrices_dir: str, shapes: dict[str, list]) -> np.ndarray:
    """
    Carrega as matrizes individuais respeitando a ordem de shapes e
    empilha em uma única matriz global (total_rows, 8000).

    A ordem de shapes determina como as linhas serão reagrupadas depois —
    é crítico que seja a mesma ordem usada no reagrupamento.
    """
    blocks = []
    for stem in shapes:
        path = os.path.join(matrices_dir, f"{stem}_matrix.npy")
        if not os.path.exists(path):
            log.error("Matriz não encontrada: %s — rode 01_kmer_matrix.py primeiro.", path)
            sys.exit(1)
        blocks.append(np.load(path))
    return np.vstack(blocks)


def plot_scree(candidates: list[int], variances: list[float],
               chosen: int, threshold: float, out_path: str) -> None:
    """Scree plot da matriz global com marcador no N escolhido."""
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


def main():
    for d in (VECTORS_DIR, REPORTS_DIR, MODELS_DIR):
        os.makedirs(d, exist_ok=True)

    for path, hint in ((SHAPES_PATH, "01_kmer_matrix.py"),
                       (IDS_PATH, "01_kmer_matrix.py")):
        if not os.path.exists(path):
            log.error("%s não encontrado — rode %s primeiro.", path, hint)
            sys.exit(1)

    with open(SHAPES_PATH) as f:
        shapes = json.load(f)   # {stem: [n_rows, 8000]}, ordem preservada

    with open(IDS_PATH) as f:
        seq_entries = json.load(f)   # [{file_stem, sequence_id}, ...]

    # ── Monta matriz global ──────────────────────────────────────────────────
    X_global = load_global_matrix(MATRICES_DIR, shapes)
    total_rows, n_features = X_global.shape
    log.info("Matriz global empilhada: %s  (%.1f MB)",
             X_global.shape, X_global.nbytes / 1e6)

    # ── Testa N candidatos ───────────────────────────────────────────────────
    max_possible = min(total_rows, n_features) - 1
    candidates = sorted(set(n for n in N_CANDIDATES if n <= max_possible) | {max_possible})
    # Remove max_possible se já estiver coberto pelos candidatos normais
    if max_possible > N_CANDIDATES[-1]:
        candidates = [n for n in N_CANDIDATES if n <= max_possible]

    log.info("Testando N candidatos: %s", candidates)
    variances: list[float] = []
    for n in candidates:
        svd_tmp = TruncatedSVD(n_components=n, random_state=RANDOM_STATE)
        svd_tmp.fit(X_global)
        cum_var = float(svd_tmp.explained_variance_ratio_.sum())
        variances.append(cum_var)
        log.info("  N=%-3d → variância acumulada: %.4f", n, cum_var)

    # ── Escolhe N ────────────────────────────────────────────────────────────
    chosen_n = candidates[-1]
    for n, var in zip(candidates, variances):
        if var >= VARIANCE_THRESHOLD:
            chosen_n = n
            break

    chosen_var = variances[candidates.index(chosen_n)]
    log.info("N escolhido: %d  (variância=%.4f)", chosen_n, chosen_var)

    plot_scree(candidates, variances, chosen_n, VARIANCE_THRESHOLD, SCREE_PLOT_PATH)

    # ── SVD final ────────────────────────────────────────────────────────────
    svd = TruncatedSVD(n_components=chosen_n, random_state=RANDOM_STATE)
    X_reduced = svd.fit_transform(X_global)   # (total_rows, chosen_n)
    log.info("X_reduced shape: %s", X_reduced.shape)

    with open(SVD_MODEL_PATH, "wb") as f:
        pickle.dump(svd, f)
    log.info("Modelo SVD salvo: %s", SVD_MODEL_PATH)

    # ── Reagrupa por proteína e aplica mean pooling ───────────────────────────
    protein_vectors: dict[str, np.ndarray] = {}
    start = 0
    for stem, (n_rows, _) in shapes.items():
        protein_rows = X_reduced[start : start + n_rows]   # (n_rows, chosen_n)
        vector = protein_rows.mean(axis=0)                  # (chosen_n,)
        protein_vectors[stem] = vector
        vec_path = os.path.join(VECTORS_DIR, f"{stem}_vector.npy")
        np.save(vec_path, vector)
        start += n_rows

        # Recupera o sequence_id legível
        seq_id = next(
            (e["sequence_id"] for e in seq_entries if e["file_stem"] == stem), stem
        )
        log.info(
            "%-30s  rows=%d  pooling: %s→%s  salvo: %s",
            seq_id[:30], n_rows, protein_rows.shape, vector.shape, vec_path,
        )

    # ── Salva sumário ─────────────────────────────────────────────────────────
    summary = {
        "n_components": chosen_n,
        "variance_captured_global": chosen_var,
        "variance_threshold": VARIANCE_THRESHOLD,
        "candidates_tested": candidates,
        "variance_per_candidate": {n: v for n, v in zip(candidates, variances)},
        "total_rows": total_rows,
        "vector_shape": [chosen_n],
        "proteins": list(shapes.keys()),
    }
    with open(SVD_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("N escolhido              : %d", chosen_n)
    log.info("Variância capturada      : %.4f", chosen_var)
    log.info("Shape do vetor por prot. : (%d,)", chosen_n)


if __name__ == "__main__":
    main()
