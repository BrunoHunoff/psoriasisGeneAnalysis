"""
Módulo 04 — Comparação das projeções SVD vs Random Projection.

Gera figura scatter 2D lado a lado e tabela CSV com distâncias ao centro
para cada proteína em ambos os espaços de projeção.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PROC_DIR            = "data/processed"
SVD_VECTORS_DIR     = os.path.join(PROC_DIR, "svd", "vectors")
RANDOM_VECTORS_DIR  = os.path.join(PROC_DIR, "random", "vectors")
SVD_SUMMARY_PATH    = os.path.join(PROC_DIR, "svd", "summary.json")
RANDOM_SUMMARY_PATH = os.path.join(PROC_DIR, "random", "summary.json")
SHAPES_PATH         = os.path.join(PROC_DIR, "matrix_shapes.json")
REPORTS_DIR         = "reports"
COMPARISON_PLOT     = os.path.join(REPORTS_DIR, "comparison_profiles.png")
COMPARISON_CSV      = os.path.join(REPORTS_DIR, "projection_comparison.csv")

SCATTER_MARKER_SIZE  = 80
CENTROID_MARKER_SIZE = 200
FIGURE_DPI           = 150


def load_vectors(vectors_dir: str, stems: list) -> dict:
    """Carrega vetores .npy para cada stem e retorna {stem: np.ndarray}."""
    vectors = {}
    for stem in stems:
        path = os.path.join(vectors_dir, f"{stem}_vector.npy")
        if not os.path.exists(path):
            log.error("Vetor não encontrado: %s", path)
            sys.exit(1)
        vectors[stem] = np.load(path)
    return vectors


def compute_distances_to_centroid(vectors: dict) -> dict:
    """Calcula a distância euclidiana de cada vetor ao centróide do conjunto."""
    matrix = np.stack(list(vectors.values()))
    centroid = matrix.mean(axis=0)
    distances = {}
    for stem, vec in vectors.items():
        distances[stem] = float(np.linalg.norm(vec - centroid))
    return distances


def plot_comparison(
    svd_vectors: dict,
    random_vectors: dict,
    stems: list,
    out_path: str,
) -> None:
    """
    Gera figura com dois scatter 2D lado a lado (SVD e Random Projection),
    com labels por proteína e centróide destacado.
    """
    colors = plt.cm.tab10(np.linspace(0, 1, len(stems)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "SVD vs Random Projection — Biomarcadores da Psoríase",
        fontsize=14, fontweight="bold",
    )

    for ax, vectors, title in (
        (axes[0], svd_vectors, "SVD"),
        (axes[1], random_vectors, "Random Projection"),
    ):
        matrix = np.stack([vectors[s] for s in stems])
        centroid = matrix.mean(axis=0)

        for i, stem in enumerate(stems):
            x, y = vectors[stem][0], vectors[stem][1]
            ax.scatter(x, y, color=colors[i], s=SCATTER_MARKER_SIZE,
                       zorder=3, label=stem)
            ax.annotate(stem, (x, y), textcoords="offset points",
                        xytext=(6, 4), fontsize=7, color=colors[i])

        ax.scatter(centroid[0], centroid[1], marker="*", s=CENTROID_MARKER_SIZE,
                   color="black", zorder=5, label="Centróide")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Componente 0", fontsize=10)
        ax.set_ylabel("Componente 1", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI)
    plt.close()
    log.info("Figura comparativa salva: %s", out_path)


def build_comparison_table(
    stems: list,
    svd_distances: dict,
    random_distances: dict,
) -> pd.DataFrame:
    """Monta DataFrame com distâncias ao centro para cada projeção."""
    rows = []
    for stem in stems:
        rows.append({
            "Proteína": stem,
            "Dist. SVD ao centro": round(svd_distances[stem], 6),
            "Dist. Random ao centro": round(random_distances[stem], 6),
        })
    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame) -> None:
    """Imprime a tabela comparativa formatada no log."""
    header = f"{'Proteína':<35} | {'Dist. SVD ao centro':>20} | {'Dist. Random ao centro':>22}"
    log.info(header)
    log.info("-" * len(header))
    for _, row in df.iterrows():
        log.info(
            "%-35s | %20.6f | %22.6f",
            row["Proteína"], row["Dist. SVD ao centro"], row["Dist. Random ao centro"],
        )


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    for path, hint in (
        (SVD_SUMMARY_PATH, "02_project_svd.py"),
        (RANDOM_SUMMARY_PATH, "03_project_random.py"),
        (SHAPES_PATH, "01_kmer_matrix.py"),
    ):
        if not os.path.exists(path):
            log.error("%s não encontrado — rode %s primeiro.", path, hint)
            sys.exit(1)

    with open(SHAPES_PATH) as f:
        shapes = json.load(f)
    stems = list(shapes.keys())

    svd_vectors = load_vectors(SVD_VECTORS_DIR, stems)
    random_vectors = load_vectors(RANDOM_VECTORS_DIR, stems)
    log.info("Vetores carregados para %d proteínas.", len(stems))

    svd_distances = compute_distances_to_centroid(svd_vectors)
    random_distances = compute_distances_to_centroid(random_vectors)

    plot_comparison(svd_vectors, random_vectors, stems, COMPARISON_PLOT)

    df = build_comparison_table(stems, svd_distances, random_distances)
    print_table(df)

    df.to_csv(COMPARISON_CSV, index=False)
    log.info("Tabela comparativa salva: %s", COMPARISON_CSV)


if __name__ == "__main__":
    main()
