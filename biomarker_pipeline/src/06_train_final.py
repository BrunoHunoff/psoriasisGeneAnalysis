"""
Módulo 06 — Treinamento final do classificador.

Treina o modelo escolhido em TODOS os dados disponíveis (sem holdout)
e salva o classificador para uso em predição de novas proteínas.

O Random Projection já foi ajustado em 02_project_random.py e está
salvo em models/random_projection.pkl — este script apenas treina
o classificador sobre os vetores X.npy resultantes.

Uso:
    python src/06_train_final.py --model svm
    python src/06_train_final.py --model logistic_regression
"""

import os
import sys
import json
import pickle
import logging
import argparse
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PROC_DIR    = "data/processed"
MODELS_DIR  = "models"
X_PATH      = os.path.join(PROC_DIR, "X.npy")
Y_PATH      = os.path.join(PROC_DIR, "y.npy")
N_COMPONENTS = 1000
RANDOM_STATE = 42


def _scaled(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


MODEL_REGISTRY = {
    "random_forest": RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE,
    ),
    "logistic_regression": _scaled(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    ),
    "svm": _scaled(
        SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    ),
    "naive_bayes": GaussianNB(),
    "knn": _scaled(KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=list(MODEL_REGISTRY.keys()),
        help="Modelo a treinar.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(MODELS_DIR, exist_ok=True)

    for path, hint in ((X_PATH, "02_project_random.py"), (Y_PATH, "02_project_random.py")):
        if not os.path.exists(path):
            log.error("%s não encontrado — rode %s primeiro.", path, hint)
            sys.exit(1)

    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    log.info("Dados: X=%s  y=%s  (pos=%d, neg=%d)",
             X.shape, y.shape, int(y.sum()), int((y == 0).sum()))

    clf = MODEL_REGISTRY[args.model]
    log.info("Treinando %s em todos os %d exemplos...", args.model, len(y))
    clf.fit(X, y)
    log.info("Treinamento concluído.")

    out_path = os.path.join(MODELS_DIR, f"classifier_{args.model}_n{N_COMPONENTS}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(clf, f)
    log.info("Classificador salvo: %s", out_path)

    meta = {
        "model": args.model,
        "n_components": N_COMPONENTS,
        "n_train": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
        "classifier_path": out_path,
        "projector_path": os.path.join(MODELS_DIR, "random_projection.pkl"),
    }
    meta_path = os.path.join(MODELS_DIR, f"classifier_{args.model}_n{N_COMPONENTS}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Metadata salvo: %s", meta_path)
    log.info("Pronto. Para predizer novas proteínas:")
    log.info("  python src/07_predict.py --model %s --fasta data/input/sua_proteina.fasta",
             args.model)


if __name__ == "__main__":
    main()
