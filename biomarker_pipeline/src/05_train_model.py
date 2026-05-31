"""
Módulo 05 — Treinamento dos modelos de one-class classification.

Três abordagens para identificar novos candidatos a biomarcador da psoríase:

1. Cosine Similarity — baseline sem treino. Compara qualquer proteína com o
   perfil médio dos 7 biomarcadores conhecidos. Referência para avaliar se
   modelos mais complexos adicionam valor.

2. One-Class SVM — aprende uma hipersuperfície que envolve os 7 vetores.
   Proteínas dentro da fronteira são candidatas. Hiperparâmetros nu e gamma
   são selecionados via leave-one-out nos 7 biomarcadores.
   Referência: Mei & Zhu (2015), Scientific Reports — One-Class SVM para
   redes de interação proteína-proteína.

3. Isolation Forest — isola pontos construindo árvores aleatórias. Pontos
   difíceis de isolar (exigem muitas partições) são normais; fáceis de isolar
   são anomalias. Biomarcadores devem ser difíceis de isolar entre si.

Por que one-class e não classificação binária:
Com 7 positivos e zero negativos rotulados, qualquer classificador binário
degeneraria. One-class learning aprende a "forma" da classe positiva e sinaliza
o que está fora — é o paradigma correto para novelty detection com poucos exemplos.
"""

import os
import sys
import json
import pickle
import logging
import argparse
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

# ── Constantes ────────────────────────────────────────────────────────────────
N_COMPONENTS   = 500
NU_VALUES      = [0.05, 0.1, 0.2]
GAMMA_VALUES   = ["scale", "auto"]
NU_DEFAULT     = 0.1
GAMMA_DEFAULT  = "scale"
N_ESTIMATORS   = 100
CONTAMINATION  = 0.1
RANDOM_STATE   = 42
PROJECTIONS    = ["svd", "random"]

PROC_DIR   = "data/processed"
MODELS_DIR = "models"
IDS_PATH   = os.path.join(PROC_DIR, "sequence_ids.json")


# ── Baseline de similaridade cosseno ─────────────────────────────────────────

class CosineSimilarityClassifier:
    """
    Baseline: calcula a similaridade cosseno de cada vetor com o perfil médio
    dos biomarcadores. Interface compatível com sklearn para avaliação uniforme.
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


# ── Carregamento de vetores ───────────────────────────────────────────────────

def load_vectors(projection: str) -> tuple[list[str], list[str], np.ndarray]:
    """
    Carrega vetores de data/processed/{projection}/vectors/.
    Retorna (stems, seq_ids, matriz (N, D)).
    """
    vecs_dir = os.path.join(PROC_DIR, projection, "vectors")
    if not os.path.isdir(vecs_dir):
        log.error(
            "Diretório de vetores não encontrado: %s — rode 04_random_projection.py primeiro.",
            vecs_dir,
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
            log.error(
                "Vetor não encontrado: %s — verifique se 04_random_projection.py rodou.", path
            )
            sys.exit(1)
        stems.append(stem)
        seq_ids.append(entry["sequence_id"])
        vectors.append(np.load(path))

    return stems, seq_ids, np.stack(vectors, axis=0)


# ── Treinamento ───────────────────────────────────────────────────────────────

def train_cosine(X: np.ndarray) -> CosineSimilarityClassifier:
    """Baseline: sem treino paramétrico — apenas calcula o centro da nuvem."""
    model = CosineSimilarityClassifier()
    model.fit(X)
    return model


def _loo_recall(nu: float, gamma: str, X: np.ndarray) -> tuple[float, float]:
    """Retorna (recall LOO, mean score LOO) para um par (nu, gamma)."""
    loo = LeaveOneOut()
    correct = 0
    scores: list[float] = []
    for train_idx, test_idx in loo.split(X):
        m = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
        m.fit(X[train_idx])
        pred = m.predict(X[test_idx])
        score = m.decision_function(X[test_idx])
        if pred[0] == 1:
            correct += 1
        scores.append(float(score[0]))
    recall = correct / len(X)
    return recall, float(np.mean(scores))


def train_ocsvm(X: np.ndarray) -> OneClassSVM:
    """
    One-Class SVM com seleção de hiperparâmetros via LOO nos 7 biomarcadores.
    Testa todas as combinações de nu e gamma, escolhe por recall máximo
    (desempate: maior score médio). Retreina no conjunto completo.
    """
    best_recall = -1.0
    best_score  = -np.inf
    best_nu, best_gamma = NU_DEFAULT, GAMMA_DEFAULT

    log.info("Seleção de hiperparâmetros OCSVM via LOO:")
    for nu in NU_VALUES:
        for gamma in GAMMA_VALUES:
            recall, mean_score = _loo_recall(nu, gamma, X)
            log.info(
                "  nu=%-4s gamma=%-5s → recall_loo=%.2f  mean_score=%.4f",
                nu, gamma, recall, mean_score,
            )
            if recall > best_recall or (recall == best_recall and mean_score > best_score):
                best_recall = recall
                best_score  = mean_score
                best_nu, best_gamma = nu, gamma

    log.info(
        "Melhor OCSVM: nu=%s gamma=%s (recall_loo=%.2f)", best_nu, best_gamma, best_recall
    )
    final = OneClassSVM(kernel="rbf", nu=best_nu, gamma=best_gamma)
    final.fit(X)
    return final


def train_isolation_forest(X: np.ndarray) -> IsolationForest:
    """Isolation Forest com hiperparâmetros fixos conforme literatura."""
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
    )
    model.fit(X)
    return model


# ── Pipeline por projeção ─────────────────────────────────────────────────────

def run_projection(projection: str) -> None:
    log.info("=== Projeção: %s ===", projection)
    stems, seq_ids, X = load_vectors(projection)
    log.info("Vetores carregados: %s  (%d proteínas × %d dims)", X.shape, *X.shape)

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Cosine (sem pkl — avaliado inline em 06)
    cosine = train_cosine(X)
    log.info("Cosine Similarity — centro calculado: shape %s", cosine.center_.shape)

    # One-Class SVM
    ocsvm = train_ocsvm(X)
    ocsvm_path = os.path.join(MODELS_DIR, f"ocsvm_{projection}.pkl")
    with open(ocsvm_path, "wb") as f:
        pickle.dump(ocsvm, f)
    log.info("OCSVM salvo: %s", ocsvm_path)

    # Isolation Forest
    iforest = train_isolation_forest(X)
    if_path = os.path.join(MODELS_DIR, f"isolation_forest_{projection}.pkl")
    with open(if_path, "wb") as f:
        pickle.dump(iforest, f)
    log.info("Isolation Forest salvo: %s", if_path)

    # Scores no conjunto de treino completo (sanidade)
    cos_scores = cosine.decision_function(X)
    ocsvm_scores = ocsvm.decision_function(X)
    if_scores = iforest.score_samples(X)
    log.info(
        "Scores treino [%s] — cosine: %.4f±%.4f | ocsvm: %.4f±%.4f | iforest: %.4f±%.4f",
        projection,
        cos_scores.mean(), cos_scores.std(),
        ocsvm_scores.mean(), ocsvm_scores.std(),
        if_scores.mean(), if_scores.std(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina modelos one-class classification.")
    parser.add_argument(
        "--projection",
        choices=["svd", "random", "both"],
        default="both",
        help="Conjunto de vetores a usar (default: both)",
    )
    args = parser.parse_args()

    targets = PROJECTIONS if args.projection == "both" else [args.projection]
    for proj in targets:
        run_projection(proj)

    log.info("Modelos treinados em: %s/", MODELS_DIR)


if __name__ == "__main__":
    main()
