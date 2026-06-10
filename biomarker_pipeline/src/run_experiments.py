"""
Script de orquestração — roda 02 + 05 para múltiplos tamanhos de vetor e modelos.

Uso:
    python src/run_experiments.py

Salva os resultados em reports/results_{model}_n{N}.json e imprime tabela final.
"""

import os
import sys
import json
import subprocess
import numpy as np

MODELS = [
    "logistic_regression",
    "svm",
    "random_forest",
    "gradient_boosting",
    "naive_bayes",
    "knn",
]

N_VALUES = [300, 400, 500, 1000, 2000]

REPORTS_DIR = "reports"
PROC_DIR    = "data/processed"


def run(cmd: list[str]) -> int:
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def read_metrics(model: str, n: int) -> dict | None:
    path = os.path.join(REPORTS_DIR, f"results_{model}_n{n}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data["summary"]


def print_table(results: dict) -> None:
    metrics = ["accuracy", "f1", "precision", "sensitivity", "specificity", "roc_auc", "mcc"]
    labels  = {
        "accuracy":    "Acurácia",
        "f1":          "F1",
        "precision":   "Precisão",
        "sensitivity": "Sensib.",
        "specificity": "Especif.",
        "roc_auc":     "ROC-AUC",
        "mcc":         "MCC",
    }

    header_models = [
        ("logistic_regression", "LR"),
        ("svm",                 "SVM"),
        ("random_forest",       "RF"),
        ("gradient_boosting",   "GB"),
        ("naive_bayes",         "NB"),
        ("knn",                 "kNN"),
    ]

    for n in N_VALUES:
        print(f"\n{'='*90}")
        print(f"  N = {n} componentes")
        print(f"{'='*90}")
        print(f"  {'Métrica':<12}", end="")
        for _, short in header_models:
            print(f"  {short:>13}", end="")
        print()
        print(f"  {'-'*12}", end="")
        for _ in header_models:
            print(f"  {'─'*13}", end="")
        print()

        for m in metrics:
            print(f"  {labels[m]:<12}", end="")
            for model, _ in header_models:
                key = (n, model)
                summary = results.get(key)
                if summary and m in summary:
                    val  = summary[m]["mean"]
                    std  = summary[m]["std"]
                    print(f"  {val*100:5.1f}±{std*100:4.1f}%", end="")
                else:
                    print(f"  {'N/A':>13}", end="")
            print()

        print()


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    results = {}

    for n in N_VALUES:
        print(f"\n{'#'*60}")
        print(f"# Gerando vetores N={n}")
        print(f"{'#'*60}")
        rc = run([sys.executable, "src/02_project_random.py", "--n-components", str(n)])
        if rc != 0:
            print(f"[ERRO] 02_project_random.py falhou para N={n}. Abortando.")
            sys.exit(1)

        for model in MODELS:
            print(f"\n  → {model}  (N={n})")
            rc = run([sys.executable, "src/05_train_evaluate.py", "--model", model])
            if rc != 0:
                print(f"  [ERRO] {model} falhou para N={n}.")
                continue

            src = os.path.join(REPORTS_DIR, f"results_{model}.json")
            dst = os.path.join(REPORTS_DIR, f"results_{model}_n{n}.json")
            if os.path.exists(src):
                with open(src) as f:
                    data = json.load(f)
                data["n_components"] = n
                with open(dst, "w") as f:
                    json.dump(data, f, indent=2)

            summary = read_metrics(model, n)
            if summary:
                results[(n, model)] = summary

    print("\n\n" + "="*90)
    print("  RESULTADOS FINAIS — StratifiedGroupKFold por proteína")
    print("="*90)
    print_table(results)


if __name__ == "__main__":
    main()
