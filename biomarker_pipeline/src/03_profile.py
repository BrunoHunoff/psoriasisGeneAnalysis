"""
Módulo 03 — Construção do perfil vetorial dos biomarcadores.

Empilha os vetores SVD das 7 proteínas em uma matriz (7, N) e calcula:
  - profile_mean: vetor médio, representa o "centro" do espaço dos biomarcadores
  - profile_std:  desvio padrão por dimensão, quantifica a dispersão

O vetor médio é o artefato central do pipeline: na etapa seguinte, qualquer
proteína candidata será projetada neste mesmo espaço e comparada por distância
coseno ou euclidiana com profile_mean.
"""

import os
import sys
import json
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PROC_DIR             = "data/processed"
VECTORS_DIR          = os.path.join(PROC_DIR, "vectors")
REPORTS_DIR          = "reports"
IDS_PATH             = os.path.join(PROC_DIR, "sequence_ids.json")
SVD_SUMMARY_PATH     = os.path.join(PROC_DIR, "svd_summary.json")
PROFILE_MEAN_PATH    = os.path.join(PROC_DIR, "profile_mean.npy")
PROFILE_STD_PATH     = os.path.join(PROC_DIR, "profile_std.npy")
PROFILE_SUMMARY_PATH = os.path.join(REPORTS_DIR, "profile_summary.json")
SCATTER_PLOT_PATH    = os.path.join(REPORTS_DIR, "biomarker_space.png")


def load_vectors(vectors_dir: str, ids_path: str) -> tuple[list[str], np.ndarray]:
    """
    Carrega todos os vetores em vectors_dir e retorna (lista de seq_ids, matriz).
    A ordem é determinada pela lista em sequence_ids.json para reprodutibilidade.
    """
    with open(ids_path) as f:
        entries = json.load(f)  # [{file_stem, sequence_id}, ...]

    seq_ids, vectors = [], []
    for entry in entries:
        stem = entry["file_stem"]
        vec_path = os.path.join(vectors_dir, f"{stem}_vector.npy")
        if not os.path.exists(vec_path):
            log.error("Vetor não encontrado: %s — rode 02_svd_per_protein.py primeiro.", vec_path)
            sys.exit(1)
        vectors.append(np.load(vec_path))
        seq_ids.append(entry["sequence_id"])

    return seq_ids, np.stack(vectors, axis=0)


def short_label(seq_id: str) -> str:
    """Extrai o nome legível de um UniProt ID como 'sp|P29460|IL12B_HUMAN'."""
    parts = seq_id.split("|")
    return parts[-1] if len(parts) >= 3 else seq_id


def plot_biomarker_space(matrix: np.ndarray, seq_ids: list[str],
                         mean_vec: np.ndarray, out_path: str) -> None:
    """
    Scatter 2D usando as dimensões 0 e 1 do vetor SVD de cada proteína.
    Inclui o vetor médio destacado como estrela vermelha.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(matrix[:, 0], matrix[:, 1],
               s=90, color="#2c7bb6", zorder=3, label="Biomarcadores")

    for i, sid in enumerate(seq_ids):
        ax.annotate(
            short_label(sid),
            (matrix[i, 0], matrix[i, 1]),
            textcoords="offset points", xytext=(6, 4), fontsize=8,
        )

    ax.scatter([mean_vec[0]], [mean_vec[1]],
               s=220, marker="*", color="#d7191c", zorder=5, label="Perfil médio")

    ax.set_xlabel("SVD componente 1", fontsize=12)
    ax.set_ylabel("SVD componente 2", fontsize=12)
    ax.set_title("Espaço SVD global — biomarcadores da psoríase", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Scatter plot salvo: %s", out_path)


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    for path, hint in ((VECTORS_DIR, "02_svd_global.py"),
                       (IDS_PATH, "01_kmer_matrix.py")):
        if not os.path.exists(path):
            log.error("%s não encontrado — rode %s primeiro.", path, hint)
            sys.exit(1)

    seq_ids, matrix = load_vectors(VECTORS_DIR, IDS_PATH)
    n_proteins, n_components = matrix.shape
    log.info("Vetores empilhados: %s  (%d proteínas × %d componentes)",
             matrix.shape, n_proteins, n_components)

    mean_vec = matrix.mean(axis=0)
    std_vec  = matrix.std(axis=0)
    norm     = float(np.linalg.norm(mean_vec))

    np.save(PROFILE_MEAN_PATH, mean_vec)
    np.save(PROFILE_STD_PATH, std_vec)

    # Distâncias de cada proteína ao perfil médio
    distances: dict[str, dict] = {}
    for sid, vec in zip(seq_ids, matrix):
        eucl = float(np.linalg.norm(vec - mean_vec))
        cos  = float(
            np.dot(vec, mean_vec) / (np.linalg.norm(vec) * np.linalg.norm(mean_vec) + 1e-12)
        )
        distances[short_label(sid)] = {"euclidean": eucl, "cosine": cos}
        log.info("  %-20s  dist_eucl=%.6f  cos_sim=%.4f", short_label(sid), eucl, cos)

    summary = {
        "n_proteins": n_proteins,
        "n_components": n_components,
        "profile_norm": norm,
        "proteins_used": [short_label(s) for s in seq_ids],
        "distances_to_mean": distances,
    }
    with open(PROFILE_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    if n_components >= 2:
        plot_biomarker_space(matrix, seq_ids, mean_vec, SCATTER_PLOT_PATH)
    else:
        log.warning("Apenas 1 componente — scatter plot 2D omitido.")

    log.info("Norma do perfil médio : %.6f", norm)
    log.info("Vetor médio (dim 0-4) : %s", np.round(mean_vec[:5], 6))
    log.info("IDs usados            : %s", [short_label(s) for s in seq_ids])
    log.info("Salvo: %s, %s, %s", PROFILE_MEAN_PATH, PROFILE_STD_PATH, PROFILE_SUMMARY_PATH)


if __name__ == "__main__":
    main()
