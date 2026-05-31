"""
Módulo 07 — Comparação dos modelos e seleção da melhor configuração.

Lê evaluation_results.json e produz:
- Figura comparativa (3 subplots): heatmap recall LOO, barras score médio,
  scatter scores individuais por proteína
- Tabela CSV com todas as métricas
- best_model_config.json com o modelo mais indicado para a busca no proteoma

Critério de seleção: maior recall_loo; desempate por maior mean_score.
"""

import os
import sys
import json
import logging
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

REPORTS_DIR      = "reports"
MODELS_DIR       = "models"
EVAL_PATH        = os.path.join(REPORTS_DIR, "evaluation_results.json")
FIGURE_PATH      = os.path.join(REPORTS_DIR, "model_comparison.png")
CSV_PATH         = os.path.join(REPORTS_DIR, "model_comparison.csv")
BEST_CONFIG_PATH = os.path.join(MODELS_DIR, "best_model_config.json")

PROJECTIONS  = ["svd", "random"]
MODEL_KEYS   = ["cosine", "ocsvm", "isolation_forest"]
MODEL_LABELS = {
    "cosine":           "Cosine Similarity",
    "ocsvm":            "One-Class SVM",
    "isolation_forest": "Isolation Forest",
}
MODEL_FILE_PREFIXES = {
    "cosine":           None,            # sem pkl próprio
    "ocsvm":            "ocsvm",
    "isolation_forest": "isolation_forest",
}
PROJ_LABELS  = {"svd": "SVD", "random": "Random Proj."}

FIGURE_DPI      = 150
FIGURE_SIZE     = (16, 5)
N_KNOWN_MARKERS = 7


def load_eval(path: str) -> dict:
    if not os.path.exists(path):
        log.error(
            "%s não encontrado — rode 06_evaluate_model.py primeiro.", path
        )
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def build_table(data: dict) -> list[dict]:
    """Constrói lista de linhas para CSV e impressão."""
    rows = []
    for model_key in MODEL_KEYS:
        for proj in PROJECTIONS:
            m = data[proj][model_key]
            rows.append({
                "model":      MODEL_LABELS[model_key],
                "projection": PROJ_LABELS[proj],
                "recall_loo": m["recall_loo"],
                "n_correct":  int(round(m["recall_loo"] * N_KNOWN_MARKERS)),
                "mean_score": m["mean_score"],
                "std_score":  m["std_score"],
                "_model_key": model_key,
                "_proj":      proj,
            })
    return rows


def find_best(rows: list[dict]) -> dict:
    """Melhor combinação: maior recall_loo; desempate por mean_score."""
    return max(rows, key=lambda r: (r["recall_loo"], r["mean_score"]))


def plot_comparison(data: dict, rows: list[dict], out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE)
    colors_proj = {"svd": "#2c7bb6", "random": "#d7191c"}

    # ── Subplot 1: Heatmap recall LOO ────────────────────────────────────────
    ax1 = axes[0]
    recall_matrix = np.array([
        [data[proj][mk]["recall_loo"] for proj in PROJECTIONS]
        for mk in MODEL_KEYS
    ])
    sns.heatmap(
        recall_matrix,
        ax=ax1,
        annot=True, fmt=".2f",
        xticklabels=[PROJ_LABELS[p] for p in PROJECTIONS],
        yticklabels=[MODEL_LABELS[mk] for mk in MODEL_KEYS],
        cmap="YlGn", vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
    )
    ax1.set_title("Recall LOO", fontsize=12, fontweight="bold")
    ax1.set_xlabel("")
    ax1.set_ylabel("")

    # ── Subplot 2: Bar plot score médio ──────────────────────────────────────
    ax2 = axes[1]
    n_models = len(MODEL_KEYS)
    x = np.arange(n_models)
    bar_width = 0.35
    for i, proj in enumerate(PROJECTIONS):
        means = [data[proj][mk]["mean_score"] for mk in MODEL_KEYS]
        stds  = [data[proj][mk]["std_score"]  for mk in MODEL_KEYS]
        ax2.bar(
            x + i * bar_width, means, bar_width,
            yerr=stds, capsize=4,
            label=PROJ_LABELS[proj],
            color=colors_proj[proj], alpha=0.85,
        )
    ax2.set_xticks(x + bar_width / 2)
    ax2.set_xticklabels([MODEL_LABELS[mk] for mk in MODEL_KEYS], fontsize=9)
    ax2.set_ylabel("Score médio de decisão")
    ax2.set_title("Score médio (modelo completo)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.grid(axis="y", alpha=0.3)

    # ── Subplot 3: Scatter scores individuais ────────────────────────────────
    ax3 = axes[2]
    markers_proj = {"svd": "o", "random": "s"}
    protein_ids = list(range(1, N_KNOWN_MARKERS + 1))
    offset_step = 0.08

    combo_list = [(mk, proj) for mk in MODEL_KEYS for proj in PROJECTIONS]
    for idx, (mk, proj) in enumerate(combo_list):
        scores = data[proj][mk]["individual_scores"]
        jitter = (idx - len(combo_list) / 2) * offset_step
        label  = f"{MODEL_LABELS[mk]} / {PROJ_LABELS[proj]}"
        ax3.scatter(
            [p + jitter for p in protein_ids], scores,
            marker=markers_proj[proj],
            color=colors_proj[proj],
            alpha=0.7 if proj == "svd" else 0.5,
            s=40,
            label=label,
        )
    ax3.set_xticks(protein_ids)
    ax3.set_xticklabels([f"P{i}" for i in protein_ids], fontsize=8)
    ax3.set_xlabel("Biomarcador")
    ax3.set_ylabel("Score de decisão")
    ax3.set_title("Scores individuais (modelo completo)", fontsize=12, fontweight="bold")
    ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax3.legend(fontsize=6, loc="best")
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI)
    plt.close()
    log.info("Figura salva: %s", out_path)


def print_table(rows: list[dict]) -> None:
    header = f"{'Modelo':<22} | {'Projeção':<12} | {'Recall LOO':>10} | {'Score médio':>12}"
    log.info("")
    log.info(header)
    log.info("-" * len(header))
    for r in rows:
        log.info(
            "%-22s | %-12s | %5d/%-4d   | %12.4f",
            r["model"], r["projection"], r["n_correct"], N_KNOWN_MARKERS, r["mean_score"],
        )
    log.info("")


def save_csv(rows: list[dict], path: str) -> None:
    fieldnames = ["model", "projection", "recall_loo", "n_correct", "mean_score", "std_score"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})
    log.info("CSV salvo: %s", path)


def save_best_config(best: dict) -> None:
    mk   = best["_model_key"]
    proj = best["_proj"]
    prefix = MODEL_FILE_PREFIXES[mk]
    model_path = (
        os.path.join(MODELS_DIR, f"{prefix}_{proj}.pkl") if prefix else None
    )
    proj_model_path = os.path.join(
        MODELS_DIR,
        "svd_global.pkl" if proj == "svd" else "random_projection.pkl",
    )
    config = {
        "model":            mk,
        "projection":       proj,
        "model_path":       model_path,
        "projection_path":  proj_model_path,
        "recall_loo":       best["recall_loo"],
        "mean_score":       best["mean_score"],
    }
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(BEST_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    log.info("Melhor config salva: %s", BEST_CONFIG_PATH)
    log.info(
        "Melhor combinação: modelo=%s  projeção=%s  recall_loo=%.2f  mean_score=%.4f",
        MODEL_LABELS[mk], PROJ_LABELS[proj], best["recall_loo"], best["mean_score"],
    )


def main() -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)

    data = load_eval(EVAL_PATH)
    rows = build_table(data)

    print_table(rows)
    save_csv(rows, CSV_PATH)
    plot_comparison(data, rows, FIGURE_PATH)

    best = find_best(rows)
    save_best_config(best)


if __name__ == "__main__":
    main()
