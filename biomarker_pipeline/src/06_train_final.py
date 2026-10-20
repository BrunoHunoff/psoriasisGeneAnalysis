"""
Módulo 06 — Treinamento final do classificador.

Treina o modelo escolhido em TODOS os dados de treino (sem holdout), com os
hiperparâmetros vencedores do GridSearchCV (05_train_evaluate.py / run_experiments.py),
e salva o classificador para predição de novas proteínas.

Representação (--k / --pattern) e N (--n-components, 0 = sem projeção) escolhem os
vetores X_n{dim}.npy e o results_{modelo}_n{dim}.json de onde vêm os hiperparâmetros.
O _meta.json registra a representação, que 07/08 usam para vetorizar sequências novas.

Pré-requisito: 02 e 05 rodados com o mesmo --dataset, representação e N
(run_experiments.py faz os dois).

Uso:
    python src/06_train_final.py --model logistic_regression --dataset tcc2
    python src/06_train_final.py --model svm --dataset tcc2 --k 4 --n-components 4000
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
from sklearn.base import clone

from common import (add_dataset_arg, add_representation_args, classifier_path, dataset_paths,
                    patterns_from_args, rep_name, results_path, vector_dim, x_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

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
    add_dataset_arg(parser)
    add_representation_args(parser)
    return parser.parse_args()


def load_best_params(reports_dir: str, model: str, n_components: int) -> dict:
    """Hiperparâmetros vencedores do GridSearchCV para (modelo, N)."""
    path = results_path(reports_dir, model, n_components)
    if os.path.exists(path):
        with open(path) as f:
            report = json.load(f)
        log.info("Hiperparâmetros de %s: %s", path, report["best_params"])
        return report["best_params"]
    log.error("Sem resultado do GridSearchCV para %s com N=%d em %s — rode antes "
              "run_experiments.py (ou 05_train_evaluate.py) com o mesmo --dataset e a mesma representação.",
              model, n_components, reports_dir)
    sys.exit(1)


def main():
    args = parse_args()
    patterns = patterns_from_args(args)
    n_components = vector_dim(patterns, args.n_components)
    paths = dataset_paths(args.dataset, patterns)
    models_dir = paths["models_dir"]
    in_x_path = x_path(paths["proc_dir"], n_components)
    y_path = os.path.join(paths["proc_dir"], "y.npy")
    os.makedirs(models_dir, exist_ok=True)

    for path in (in_x_path, y_path):
        if not os.path.exists(path):
            log.error("%s não encontrado — rode 02_project_random.py --dataset %s com a mesma "
                      "representação e --n-components %d primeiro.", path, args.dataset, args.n_components)
            sys.exit(1)

    X = np.load(in_x_path)
    y = np.load(y_path)
    log.info("Dataset %s, representação %s — X=%s  y=%s  (pos=%d, neg=%d)",
             args.dataset, rep_name(patterns), X.shape, y.shape, int(y.sum()), int((y == 0).sum()))

    best_params = load_best_params(paths["reports_dir"], args.model, n_components)
    clf = clone(MODEL_REGISTRY[args.model]).set_params(**best_params)
    log.info("Treinando %s em todos os %d exemplos...", args.model, len(y))
    clf.fit(X, y)
    log.info("Treinamento concluído.")

    out_path = classifier_path(models_dir, args.model, n_components)
    with open(out_path, "wb") as f:
        pickle.dump(clf, f)
    log.info("Classificador salvo: %s", out_path)

    meta = {
        "model": args.model,
        "dataset": args.dataset,
        "representation": rep_name(patterns),
        "patterns": list(patterns),
        "projection": bool(args.n_components),
        "random_state": RANDOM_STATE,
        "n_components": n_components,
        "best_params": best_params,
        "n_train": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
        "classifier_path": out_path,
    }
    meta_path = out_path.replace(".pkl", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Metadata salvo: %s", meta_path)
    log.info("Pronto. Para predizer novas proteínas:")
    rep_args = " ".join(["--pattern", *patterns] if args.pattern else ["--k", str(args.k)])
    log.info("  python src/07_predict.py --model %s --dataset %s %s --n-components %d "
             "--fasta data/input/sua_proteina.fasta", args.model, args.dataset, rep_args, args.n_components)


if __name__ == "__main__":
    main()
