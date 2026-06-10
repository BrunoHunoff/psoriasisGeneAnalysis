"""
Módulo 05 — Treinamento e avaliação com cross-validation.

Uso:
    python src/05_train_evaluate.py --model random_forest
    python src/05_train_evaluate.py --model logistic_regression
    python src/05_train_evaluate.py --model svm
    python src/05_train_evaluate.py --model naive_bayes
    python src/05_train_evaluate.py --model knn
    python src/05_train_evaluate.py --model gradient_boosting

Para adicionar um novo modelo: inclua uma entrada em MODEL_REGISTRY.
Modelos sensíveis à escala (LR, SVM, kNN) já estão envolvidos em
Pipeline(StandardScaler, modelo) — o scaler é ajustado dentro de cada
fold, garantindo que dados de teste nunca vazam para o treino.

Usa StratifiedGroupKFold agrupando por proteína (campo "protein" do
metadata.json), garantindo que ortólogos/isoformas da mesma proteína
nunca apareçam simultaneamente em treino e teste.

Métricas por fold (média ± desvio padrão ao final):
    Accuracy, F1, Precisão, Sensibilidade, Especificidade, ROC-AUC, MCC
"""

import os
import sys
import json
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PROC_DIR      = "data/processed"
REPORTS_DIR   = "reports"
X_PATH        = os.path.join(PROC_DIR, "X.npy")
Y_PATH        = os.path.join(PROC_DIR, "y.npy")
METADATA_PATH = os.path.join(PROC_DIR, "metadata.json")

N_FOLDS      = 10
RANDOM_STATE = 42

# ── Adicione novos modelos aqui ──────────────────────────────────────────────
def _scaled(model):
    """Envolve um estimador em Pipeline com StandardScaler."""
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


MODEL_REGISTRY = {
    "random_forest": RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=RANDOM_STATE,
    ),
    "logistic_regression": _scaled(
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
    ),
    "svm": _scaled(
        SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE,
        )
    ),
    "naive_bayes": GaussianNB(),
    "knn": _scaled(
        KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1,
        )
    ),
}
# ────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parseia argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description="Treina e avalia um modelo via cross-validation.")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Nome do modelo a usar (definido em MODEL_REGISTRY).",
    )
    return parser.parse_args()


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula especificidade (True Negative Rate = TN / (TN + FP))."""
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def evaluate_fold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Calcula todas as métricas para um único fold."""
    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "f1":          f1_score(y_true, y_pred, zero_division=0),
        "precision":   precision_score(y_true, y_pred, zero_division=0),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity_score(y_true, y_pred),
        "roc_auc":     roc_auc_score(y_true, y_prob),
        "mcc":         matthews_corrcoef(y_true, y_pred),
    }


def summarize(fold_results: list[dict]) -> dict:
    """Calcula média e desvio padrão de cada métrica sobre todos os folds."""
    metrics = list(fold_results[0].keys())
    summary = {}
    for m in metrics:
        values = [r[m] for r in fold_results]
        summary[m] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return summary


def print_summary(model_name: str, summary: dict, n_folds: int) -> None:
    """Imprime tabela de resultados no log."""
    log.info("=" * 55)
    log.info("Modelo: %s  (%d-fold CV — GroupKFold por proteína)", model_name, n_folds)
    log.info("=" * 55)
    log.info("%-16s  %8s  %8s", "Métrica", "Média", "Desvio")
    log.info("-" * 40)
    labels = {
        "accuracy":    "Acurácia",
        "f1":          "F1 Score",
        "precision":   "Precisão",
        "sensitivity": "Sensibilidade",
        "specificity": "Especificidade",
        "roc_auc":     "ROC-AUC",
        "mcc":         "MCC",
    }
    for key, label in labels.items():
        m = summary[key]
        log.info("%-16s  %8.4f  %8.4f", label, m["mean"], m["std"])
    log.info("=" * 55)


def main() -> None:
    args = parse_args()
    os.makedirs(REPORTS_DIR, exist_ok=True)

    for path, hint in ((X_PATH, "02_project_random.py"), (Y_PATH, "02_project_random.py")):
        if not os.path.exists(path):
            log.error("%s não encontrado — rode %s primeiro.", path, hint)
            sys.exit(1)

    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    if not os.path.exists(METADATA_PATH):
        log.error("%s não encontrado — rode 02_project_random.py primeiro.", METADATA_PATH)
        sys.exit(1)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    groups = np.array([metadata[str(i)]["protein"] for i in range(len(y))])
    n_unique_groups = len(set(groups))

    log.info("Dados carregados: X=%s  y=%s  (pos=%d, neg=%d)",
             X.shape, y.shape, int(y.sum()), int((y == 0).sum()))
    log.info("Grupos por proteína: %d proteínas distintas", n_unique_groups)

    n_folds = min(N_FOLDS, n_unique_groups)
    if n_folds < N_FOLDS:
        log.warning(
            "Número de proteínas distintas (%d) < N_FOLDS (%d). "
            "Usando %d folds.", n_unique_groups, N_FOLDS, n_folds,
        )

    model = MODEL_REGISTRY[args.model]
    log.info("Modelo selecionado: %s", args.model)
    log.info("Configuração: %s", model)

    skf = StratifiedGroupKFold(n_splits=n_folds)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y, groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = evaluate_fold(y_test, y_pred, y_prob)
        fold_results.append(metrics)
        log.info(
            "Fold %2d/%d — acc=%.4f  f1=%.4f  auc=%.4f  mcc=%.4f",
            fold, n_folds,
            metrics["accuracy"], metrics["f1"],
            metrics["roc_auc"], metrics["mcc"],
        )

    summary = summarize(fold_results)
    print_summary(args.model, summary, n_folds)

    report = {
        "model": args.model,
        "cv_strategy": "StratifiedGroupKFold (agrupado por proteína)",
        "n_folds": n_folds,
        "n_unique_proteins": n_unique_groups,
        "n_samples": int(X.shape[0]),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
        "fold_results": fold_results,
        "summary": summary,
    }
    out_path = os.path.join(REPORTS_DIR, f"results_{args.model}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Resultados salvos: %s", out_path)


if __name__ == "__main__":
    main()
