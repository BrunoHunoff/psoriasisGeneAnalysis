"""
Módulo 06 — Treinamento final do classificador.

Treina o modelo escolhido em TODOS os dados de treino (sem holdout), com os
hiperparâmetros vencedores do GridSearchCV (05_train_evaluate.py / run_experiments.py),
e salva o classificador para predição de novas proteínas.

N vem dos próprios vetores (X.npy gerado por 02_project_random.py), então o nome do
classificador sempre corresponde ao projetor salvo em {models_dir}/random_projection.pkl.

Pré-requisito: 02 e 05 rodados com o mesmo --dataset e o mesmo N (run_experiments.py
termina em N=1000; para outro N, rode 02 --n-components N e 05 de novo).

Uso:
    python src/06_train_final.py --model logistic_regression --dataset tcc2
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

from common import add_dataset_arg, classifier_path, dataset_paths, projector_path

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
    return parser.parse_args()


def load_best_params(reports_dir: str, model: str, n_components: int) -> dict:
    """Hiperparâmetros vencedores do GridSearchCV para (modelo, N)."""
    candidates = [os.path.join(reports_dir, f"results_{model}_n{n_components}.json"),
                  os.path.join(reports_dir, f"results_{model}.json")]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                report = json.load(f)
            if report.get("n_components") == n_components:
                log.info("Hiperparâmetros de %s: %s", path, report["best_params"])
                return report["best_params"]
    log.error("Sem resultado do GridSearchCV para %s com N=%d em %s — rode antes "
              "run_experiments.py (ou 05_train_evaluate.py) com o mesmo --dataset.",
              model, n_components, reports_dir)
    sys.exit(1)


def main():
    args = parse_args()
    paths = dataset_paths(args.dataset)
    models_dir = paths["models_dir"]
    x_path = os.path.join(paths["proc_dir"], "X.npy")
    y_path = os.path.join(paths["proc_dir"], "y.npy")
    os.makedirs(models_dir, exist_ok=True)

    for path in (x_path, y_path):
        if not os.path.exists(path):
            log.error("%s não encontrado — rode 02_project_random.py --dataset %s primeiro.",
                      path, args.dataset)
            sys.exit(1)

    X = np.load(x_path)
    y = np.load(y_path)
    n_components = X.shape[1]
    log.info("Dataset %s — X=%s  y=%s  (pos=%d, neg=%d)",
             args.dataset, X.shape, y.shape, int(y.sum()), int((y == 0).sum()))

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
        "n_components": n_components,
        "best_params": best_params,
        "n_train": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
        "classifier_path": out_path,
        "projector_path": projector_path(models_dir),
    }
    meta_path = out_path.replace(".pkl", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Metadata salvo: %s", meta_path)
    log.info("Pronto. Para predizer novas proteínas:")
    log.info("  python src/07_predict.py --model %s --dataset %s --fasta data/input/sua_proteina.fasta",
             args.model, args.dataset)


if __name__ == "__main__":
    main()
