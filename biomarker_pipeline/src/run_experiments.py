"""
Script de orquestração — roda 02 + 05 para múltiplos tamanhos de vetor e modelos.

Uso:
    python src/run_experiments.py --dataset tcc2

Salva resultados em {reports_dir}/results_{model}_n{N}.json e gera tabelas
em {reports_dir}/tabelas_resultados.tsv (copiar e colar no Google Docs).
reports_dir = reports/ (tcc1) ou reports/v2/ (tcc2) — ver common.DATASETS.

O último N de N_VALUES (1000) é o que fica salvo em random_projection.pkl e nos
classificadores — é a configuração usada por 06/07/08.
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import date

from common import add_dataset_arg, dataset_paths

MODELS = [
    "logistic_regression",
    "svm",
    "random_forest",
    "gradient_boosting",
    "naive_bayes",
    "knn",
]

MODEL_LABELS = {
    "logistic_regression": "Regressão Logística",
    "svm":                 "SVM",
    "random_forest":       "Random Forest",
    "gradient_boosting":   "Gradient Boosting",
    "naive_bayes":         "Naive Bayes",
    "knn":                 "kNN",
}

METRIC_LABELS = {
    "accuracy":    "Acurácia",
    "f1":          "F1 Score",
    "precision":   "Precisão",
    "sensitivity": "Sensibilidade",
    "specificity": "Especificidade",
    "roc_auc":     "ROC-AUC",
    "mcc":         "MCC",
}

N_VALUES    = [300, 400, 500, 1000]   # manter 1000 por último (configuração final)


def run(cmd: list[str]) -> int:
    return subprocess.run(cmd).returncode


def save_result(reports_dir: str, model: str, n: int) -> dict | None:
    src = os.path.join(reports_dir, f"results_{model}.json")
    dst = os.path.join(reports_dir, f"results_{model}_n{n}.json")
    if not os.path.exists(src):
        return None
    with open(src) as f:
        data = json.load(f)
    data["n_components"] = n
    with open(dst, "w") as f:
        json.dump(data, f, indent=2)
    return data.get("metrics")


def load_result(reports_dir: str, model: str, n: int) -> dict | None:
    path = os.path.join(reports_dir, f"results_{model}_n{n}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get("metrics")


def fmt(mean: float, std: float) -> str:
    return f"{mean*100:.1f}% ± {std*100:.1f}%"


def build_tsv(results: dict, dataset: str) -> str:
    lines = []
    ano = date.today().year
    table_num = 0

    for n in N_VALUES:
        table_num += 1
        lines.append(
            f"Tabela {table_num} — Desempenho dos classificadores com N={n} componentes "
            f"(dataset {dataset}; média ± desvio padrão, {10}-fold cross-validation com GridSearchCV)"
        )
        lines.append("")

        header = ["Modelo"] + list(METRIC_LABELS.values())
        lines.append("\t".join(header))

        for model in MODELS:
            metrics = results.get((n, model))
            if metrics is None:
                row = [MODEL_LABELS[model]] + ["N/D"] * len(METRIC_LABELS)
            else:
                row = [MODEL_LABELS[model]]
                for metric in METRIC_LABELS:
                    m = metrics.get(metric, {})
                    row.append(fmt(m.get("mean", 0), m.get("std", 0)))
            lines.append("\t".join(row))

        lines.append(f"Fonte: Elaborado pelos autores ({ano}).")
        lines.append("")
        lines.append("")

    return "\n".join(lines)


def print_terminal(results: dict) -> None:
    for n in N_VALUES:
        print(f"\n{'='*100}")
        print(f"  N = {n} componentes")
        print(f"{'='*100}")
        col = 20
        print(f"  {'Modelo':<25}", end="")
        for label in METRIC_LABELS.values():
            print(f"  {label:>{col}}", end="")
        print()
        print(f"  {'-'*25}", end="")
        for _ in METRIC_LABELS:
            print(f"  {'─'*col}", end="")
        print()

        for model in MODELS:
            metrics = results.get((n, model))
            print(f"  {MODEL_LABELS[model]:<25}", end="")
            for metric in METRIC_LABELS:
                if metrics and metric in metrics:
                    m = metrics[metric]
                    cell = fmt(m["mean"], m["std"])
                else:
                    cell = "N/D"
                print(f"  {cell:>{col}}", end="")
            print()


def main():
    parser = argparse.ArgumentParser()
    add_dataset_arg(parser)
    args = parser.parse_args()
    reports_dir = dataset_paths(args.dataset)["reports_dir"]
    tsv_path = os.path.join(reports_dir, "tabelas_resultados.tsv")
    os.makedirs(reports_dir, exist_ok=True)
    results = {}
    dataset_arg = ["--dataset", args.dataset]

    for n in N_VALUES:
        print(f"\n{'#'*60}")
        print(f"# Gerando vetores N={n}")
        print(f"{'#'*60}")
        rc = run([sys.executable, "src/02_project_random.py", "--n-components", str(n)] + dataset_arg)
        if rc != 0:
            print(f"[ERRO] 02_project_random.py falhou para N={n}.")
            sys.exit(1)

        for model in MODELS:
            print(f"\n  → {model}  (N={n})")
            rc = run([sys.executable, "src/05_train_evaluate.py", "--model", model] + dataset_arg)
            if rc != 0:
                print(f"  [ERRO] {model} falhou para N={n}.")
                continue
            metrics = save_result(reports_dir, model, n)
            if metrics:
                results[(n, model)] = metrics

    print_terminal(results)

    tsv = build_tsv(results, args.dataset)
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write(tsv)

    print(f"\n{'='*60}")
    print(f"  Tabelas salvas em: {tsv_path}")
    print(f"  Para usar no Google Docs:")
    print(f"    1. Abra o arquivo {tsv_path}")
    print(f"    2. Selecione todo o conteúdo de uma tabela")
    print(f"    3. Cole no Google Docs — ele cria a tabela automaticamente")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
