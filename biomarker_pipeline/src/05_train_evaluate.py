"""
Módulo 05 — Avaliação e treinamento com GridSearchCV.

Usa GridSearchCV (10-fold StratifiedKFold) para encontrar os melhores
hiperparâmetros de cada modelo e reportar as métricas de desempenho, nos vetores
X_n{dim}.npy gerados pelo 02 com a mesma representação e o mesmo --n-components.
Resultados em {reports_dir}/results_{modelo}_n{dim}.json; o melhor modelo é salvo em
{models_dir}/classifier_{modelo}_n{dim}.pkl.

Uso:
    python src/05_train_evaluate.py --model random_forest --dataset tcc2
    python src/05_train_evaluate.py --model svm --dataset tcc2 --k 4 --n-components 4000
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    make_scorer,
    matthews_corrcoef,
    confusion_matrix,
)

from common import (add_dataset_arg, add_representation_args, classifier_path, dataset_paths,
                    patterns_from_args, rep_name, results_path, vector_dim, x_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

N_FOLDS      = 10
RANDOM_STATE = 42


def _scaled(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


MODEL_REGISTRY = {
    "logistic_regression": _scaled(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    "svm":                 _scaled(SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
    "random_forest":       RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "gradient_boosting":   GradientBoostingClassifier(random_state=RANDOM_STATE),
    "naive_bayes":         GaussianNB(),
    "knn":                 _scaled(KNeighborsClassifier(n_jobs=-1)),
}

PARAM_GRIDS = {
    "logistic_regression": {"model__C": [0.01, 0.1, 1, 10, 100]},
    "svm":                 {"model__C": [0.1, 1, 10], "model__gamma": ["scale", "auto"]},
    "random_forest":       {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
    "gradient_boosting":   {"n_estimators": [50, 100, 200], "learning_rate": [0.05, 0.1, 0.2]},
    "naive_bayes":         {"var_smoothing": [1e-9, 1e-7, 1e-5]},
    "knn":                 {"model__n_neighbors": [3, 5, 7, 11]},
}


def _specificity(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


SCORING = {
    "accuracy":    "accuracy",
    "f1":          "f1",
    "precision":   "precision",
    "sensitivity": "recall",
    "specificity": make_scorer(_specificity),
    "roc_auc":     "roc_auc",
    "mcc":         make_scorer(matthews_corrcoef),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    add_dataset_arg(parser)
    add_representation_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    patterns = patterns_from_args(args)
    n_components = vector_dim(patterns, args.n_components)
    paths = dataset_paths(args.dataset, patterns)
    reports_dir, models_dir = paths["reports_dir"], paths["models_dir"]
    in_x_path = x_path(paths["proc_dir"], n_components)
    y_path = os.path.join(paths["proc_dir"], "y.npy")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    for path in (in_x_path, y_path):
        if not os.path.exists(path):
            log.error("%s não encontrado — rode 02_project_random.py --dataset %s com a mesma "
                      "representação e --n-components %d primeiro.", path, args.dataset, args.n_components)
            sys.exit(1)

    X = np.load(in_x_path)
    y = np.load(y_path)
    log.info("Dataset %s, representação %s — X=%s  y=%s  (pos=%d, neg=%d)  N=%d%s",
             args.dataset, rep_name(patterns), X.shape, y.shape, int(y.sum()), int((y == 0).sum()),
             n_components, "" if args.n_components else " (sem projeção)")

    model      = MODEL_REGISTRY[args.model]
    param_grid = PARAM_GRIDS[args.model]
    cv         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    n_combinations = 1
    for v in param_grid.values():
        n_combinations *= len(v)
    log.info("GridSearchCV: %d combinações × %d folds = %d fits",
             n_combinations, N_FOLDS, n_combinations * N_FOLDS)

    grid = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=SCORING,
        refit="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X, y)

    best_idx = grid.best_index_
    metrics = {}
    for metric in SCORING:
        metrics[metric] = {
            "mean": float(grid.cv_results_[f"mean_test_{metric}"][best_idx]),
            "std":  float(grid.cv_results_[f"std_test_{metric}"][best_idx]),
        }

    log.info("Melhores parâmetros: %s", grid.best_params_)
    log.info("ROC-AUC: %.4f ± %.4f", metrics["roc_auc"]["mean"], metrics["roc_auc"]["std"])

    clf_path = classifier_path(models_dir, args.model, n_components)
    with open(clf_path, "wb") as f:
        pickle.dump(grid.best_estimator_, f)
    log.info("Melhor modelo salvo: %s", clf_path)

    report = {
        "model":        args.model,
        "dataset":      args.dataset,
        "representation": rep_name(patterns),
        "patterns":     list(patterns),
        "projection":   bool(args.n_components),
        "n_components": n_components,
        "n_folds":      N_FOLDS,
        "n_samples":    int(X.shape[0]),
        "n_positive":   int(y.sum()),
        "n_negative":   int((y == 0).sum()),
        "best_params":  grid.best_params_,
        "metrics":      metrics,
    }
    out_path = results_path(reports_dir, args.model, n_components)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Resultados salvos: %s", out_path)


if __name__ == "__main__":
    main()
