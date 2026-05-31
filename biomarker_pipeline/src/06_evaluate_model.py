"""
Módulo 06 — Avaliação dos modelos via Leave-One-Out.

Como não existem negativos rotulados, o protocolo de avaliação é o teste
de sanidade (leave-one-out): para cada biomarcador, treina com os outros 6
e verifica se o modelo identifica o deixado de fora como positivo.

Métricas reportadas por modelo e projeção:
- recall_loo      : fração dos 7 marcadores corretamente identificados como inliers
- mean_score      : score médio do modelo treinado em TODOS os 7 (confiança geral)
- std_score       : desvio padrão dos scores (coesão interna da classe)
- individual_scores: scores de cada proteína no modelo completo (para scatter em 07)

A avaliação LOO reusa os mesmos hiperparâmetros do modelo salvo em 05,
garantindo que o protocolo de seleção não vaze para a avaliação.
"""

import os
import sys
import json
import pickle
import logging
import numpy as np

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import LeaveOneOut

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

RANDOM_STATE = 42
PROJECTIONS  = ["svd", "random"]
PROC_DIR     = "data/processed"
MODELS_DIR   = "models"
REPORTS_DIR  = "reports"
IDS_PATH     = os.path.join(PROC_DIR, "sequence_ids.json")
EVAL_PATH    = os.path.join(REPORTS_DIR, "evaluation_results.json")


# ── Baseline ──────────────────────────────────────────────────────────────────

class CosineSimilarityClassifier:
    """
    Baseline: compara com o perfil médio por similaridade cosseno.
    Interface sklearn para avaliação uniforme com OC-SVM e Isolation Forest.
    """

    def __init__(self) -> None:
        self.center_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "CosineSimilarityClassifier":
        self.center_ = X.mean(axis=0, keepdims=True)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return cosine_similarity(X, self.center_).ravel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) > 0, 1, -1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return self.decision_function(X)


# ── Carregamento ──────────────────────────────────────────────────────────────

def load_vectors(projection: str) -> tuple[list[str], list[str], np.ndarray]:
    """Carrega vetores de data/processed/{projection}/vectors/."""
    vecs_dir = os.path.join(PROC_DIR, projection, "vectors")
    if not os.path.isdir(vecs_dir):
        log.error(
            "Diretório não encontrado: %s — rode 04_random_projection.py primeiro.", vecs_dir
        )
        sys.exit(1)
    if not os.path.exists(IDS_PATH):
        log.error("%s não encontrado — rode 01_kmer_matrix.py primeiro.", IDS_PATH)
        sys.exit(1)
    with open(IDS_PATH) as f:
        entries = json.load(f)
    stems, seq_ids, vectors = [], [], []
    for entry in entries:
        stem = entry["file_stem"]
        path = os.path.join(vecs_dir, f"{stem}_vector.npy")
        if not os.path.exists(path):
            log.error("Vetor não encontrado: %s", path)
            sys.exit(1)
        stems.append(stem)
        seq_ids.append(entry["sequence_id"])
        vectors.append(np.load(path))
    return stems, seq_ids, np.stack(vectors, axis=0)


def load_model(path: str):
    """Carrega modelo pickle; aborta com mensagem clara se ausente."""
    if not os.path.exists(path):
        log.error("Modelo não encontrado: %s — rode 05_train_model.py primeiro.", path)
        sys.exit(1)
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Helpers de score ──────────────────────────────────────────────────────────

def get_scores(model, X: np.ndarray) -> np.ndarray:
    """
    Scores normalizados: maior = mais parecido com biomarcador.
    - OC-SVM / Cosine: decision_function
    - Isolation Forest: score_samples
    """
    if isinstance(model, IsolationForest):
        return model.score_samples(X)
    return model.decision_function(X)


def loo_evaluate(model_factory, X: np.ndarray) -> tuple[float, list[float]]:
    """Avalia via LOO. Retorna (recall, scores_loo por proteína)."""
    loo = LeaveOneOut()
    correct = 0
    loo_scores: list[float] = []
    for train_idx, test_idx in loo.split(X):
        m = model_factory()
        m.fit(X[train_idx])
        pred = m.predict(X[test_idx])
        score = get_scores(m, X[test_idx])
        if pred[0] == 1:
            correct += 1
        loo_scores.append(float(score[0]))
    return correct / len(X), loo_scores


# ── Avaliação por projeção ────────────────────────────────────────────────────

def evaluate_projection(projection: str, X: np.ndarray) -> dict:
    """Avalia as 3 abordagens para uma projeção. Retorna dict de métricas."""
    results: dict[str, dict] = {}

    # ── Cosine ───────────────────────────────────────────────────────────────
    full_cos = CosineSimilarityClassifier().fit(X)
    full_scores = get_scores(full_cos, X)
    recall, _ = loo_evaluate(lambda: CosineSimilarityClassifier(), X)
    results["cosine"] = {
        "recall_loo":       round(recall, 4),
        "mean_score":       round(float(full_scores.mean()), 6),
        "std_score":        round(float(full_scores.std()), 6),
        "individual_scores": [round(float(s), 6) for s in full_scores],
    }
    log.info(
        "[%s] cosine   recall_loo=%.2f  mean_score=%.4f  std=%.4f",
        projection, recall, full_scores.mean(), full_scores.std(),
    )

    # ── One-Class SVM ────────────────────────────────────────────────────────
    ocsvm = load_model(os.path.join(MODELS_DIR, f"ocsvm_{projection}.pkl"))
    full_scores = get_scores(ocsvm, X)
    nu, gamma, kernel = ocsvm.nu, ocsvm.gamma, ocsvm.kernel
    recall, _ = loo_evaluate(
        lambda: OneClassSVM(kernel=kernel, nu=nu, gamma=gamma), X
    )
    results["ocsvm"] = {
        "recall_loo":       round(recall, 4),
        "mean_score":       round(float(full_scores.mean()), 6),
        "std_score":        round(float(full_scores.std()), 6),
        "individual_scores": [round(float(s), 6) for s in full_scores],
        "params":           {"nu": nu, "gamma": str(gamma), "kernel": kernel},
    }
    log.info(
        "[%s] ocsvm    recall_loo=%.2f  mean_score=%.4f  std=%.4f  (nu=%s gamma=%s)",
        projection, recall, full_scores.mean(), full_scores.std(), nu, gamma,
    )

    # ── Isolation Forest ─────────────────────────────────────────────────────
    iforest = load_model(os.path.join(MODELS_DIR, f"isolation_forest_{projection}.pkl"))
    full_scores = get_scores(iforest, X)
    n_est, contam = iforest.n_estimators, iforest.contamination
    recall, _ = loo_evaluate(
        lambda: IsolationForest(
            n_estimators=n_est, contamination=contam, random_state=RANDOM_STATE
        ),
        X,
    )
    results["isolation_forest"] = {
        "recall_loo":       round(recall, 4),
        "mean_score":       round(float(full_scores.mean()), 6),
        "std_score":        round(float(full_scores.std()), 6),
        "individual_scores": [round(float(s), 6) for s in full_scores],
        "params":           {"n_estimators": n_est, "contamination": contam},
    }
    log.info(
        "[%s] iforest  recall_loo=%.2f  mean_score=%.4f  std=%.4f",
        projection, recall, full_scores.mean(), full_scores.std(),
    )

    return results


def main() -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)

    all_results: dict[str, dict] = {}
    for proj in PROJECTIONS:
        stems, seq_ids, X = load_vectors(proj)
        log.info("=== Avaliando: %s (%d proteínas × %d dims) ===", proj, *X.shape)
        all_results[proj] = evaluate_projection(proj, X)

    with open(EVAL_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Resultados salvos: %s", EVAL_PATH)


if __name__ == "__main__":
    main()
