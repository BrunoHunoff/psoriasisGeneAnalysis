"""
Script de orquestração — roda 01 + 02 + 05 para uma representação k-mer, vários tamanhos
de vetor e vários modelos.

Uso:
    python src/run_experiments.py --dataset tcc2
    python src/run_experiments.py --dataset tcc2 --n-values 2000 4000 0
    python src/run_experiments.py --dataset tcc2 --k 2 --n-values 0
    python src/run_experiments.py --dataset tcc2 --pattern 1101 1011 --n-values 1000 4000 0
    python src/run_experiments.py --dataset tcc2 --k 4 --n-values 0 --models svm logistic_regression
    python src/run_experiments.py --dataset tcc2 --resumo     # só refaz as tabelas

--n-values 0 = sem projeção (vetor de frequências dos k-mers; ver common.kmer_vectors).

Salva resultados em {reports_dir}/results_{model}_n{dim}.json (escritos pelo 05) e gera:
  {reports_dir}/tabelas_resultados.tsv — uma tabela por N, com todos os N já rodados
      na representação (não só os desta execução)
  {base}/tabelas_representacoes.tsv    — ROC-AUC e MCC de todas as representações × N,
      para escolher k e N pela CV
reports_dir = reports/v2/ para k=3 contínuo no tcc2, reports/v2/{representação}/ para as
demais (ver common.dataset_paths); base = reports/v2/ (tcc2) ou reports/ (tcc1).
"""

import os
import re
import sys
import json
import argparse
import subprocess
from datetime import date

from common import (REP_NAME_RE, add_dataset_arg, add_representation_args, dataset_paths,
                    patterns_from_args, rep_name)

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

SUMMARY_METRICS = ("roc_auc", "mcc")

N_VALUES    = [300, 400, 500, 1000]   # linha de base do TCC I/II; 0 = sem projeção
RESULT_RE   = re.compile(r"^results_(.+)_n(\d+)\.json$")


def run(cmd: list[str]) -> int:
    return subprocess.run(cmd).returncode


def load_results(reports_dir: str) -> dict:
    """{(dim, modelo): relatório} de todos os results_{modelo}_n{dim}.json do diretório."""
    results = {}
    if not os.path.isdir(reports_dir):
        return results
    for fname in os.listdir(reports_dir):
        m = RESULT_RE.match(fname)
        if m and m.group(1) in MODELS:
            with open(os.path.join(reports_dir, fname)) as f:
                results[(int(m.group(2)), m.group(1))] = json.load(f)
    return results


def rep_dirs(base_reports_dir: str) -> dict[str, str]:
    """{representação: diretório de relatórios} — k3 na base, as demais em subdiretórios."""
    dirs = {"k3": base_reports_dir}
    for name in os.listdir(base_reports_dir):
        if name != "k3" and REP_NAME_RE.match(name) and os.path.isdir(os.path.join(base_reports_dir, name)):
            dirs[name] = os.path.join(base_reports_dir, name)
    return dict(sorted(dirs.items(), key=lambda kv: (int(re.match(r"k(\d+)", kv[0]).group(1)), kv[0])))


def n_label(dim: int, report: dict) -> str:
    return str(dim) if report.get("projection", True) else f"sem projeção ({dim})"


def rep_label(rep: str) -> str:
    """k3 → k=3; k3_1101+1011 → k=3 (1101+1011)."""
    k, _, patterns = rep.partition("_")
    return f"k={k[1:]}" + (f" ({patterns})" if patterns else "")


def fmt(mean: float, std: float) -> str:
    return f"{mean*100:.1f}% ± {std*100:.1f}%"


def dims_of(results: dict) -> list[int]:
    return sorted({dim for dim, _ in results})


def build_tsv(results: dict, dataset: str, rep: str) -> str:
    lines = []
    ano = date.today().year
    table_num = 0

    for n in dims_of(results):
        table_num += 1
        any_report = next(r for (d, _), r in results.items() if d == n)
        title_n = (f"N={n} componentes" if any_report.get("projection", True)
                   else f"vetor sem projeção ({n} k-mers)")
        lines.append(
            f"Tabela {table_num} — Desempenho dos classificadores com {title_n} "
            f"(dataset {dataset}, representação {rep_label(rep)}; média ± desvio padrão, "
            f"{10}-fold cross-validation com GridSearchCV)"
        )
        lines.append("")

        header = ["Modelo"] + list(METRIC_LABELS.values())
        lines.append("\t".join(header))

        for model in MODELS:
            report = results.get((n, model))
            if report is None:
                row = [MODEL_LABELS[model]] + ["N/D"] * len(METRIC_LABELS)
            else:
                row = [MODEL_LABELS[model]]
                for metric in METRIC_LABELS:
                    m = report["metrics"].get(metric, {})
                    row.append(fmt(m.get("mean", 0), m.get("std", 0)))
            lines.append("\t".join(row))

        lines.append(f"Fonte: Elaborado pelos autores ({ano}).")
        lines.append("")
        lines.append("")

    return "\n".join(lines)


def build_summary_tsv(all_results: dict, dataset: str) -> str:
    """Uma tabela por métrica: linhas = (representação, N), colunas = modelos."""
    lines = []
    ano = date.today().year
    for metric in SUMMARY_METRICS:
        lines.append(
            f"Tabela — {METRIC_LABELS[metric]} por representação k-mer e tamanho do vetor "
            f"(dataset {dataset}; média ± desvio padrão, 10-fold cross-validation com GridSearchCV)"
        )
        lines.append("")
        lines.append("\t".join(["Representação", "N"] + [MODEL_LABELS[m] for m in MODELS]))
        for rep, results in all_results.items():
            for n in dims_of(results):
                any_report = next(r for (d, _), r in results.items() if d == n)
                row = [rep_label(rep), n_label(n, any_report)]
                for model in MODELS:
                    report = results.get((n, model))
                    if report is None:
                        row.append("N/D")
                    else:
                        m = report["metrics"][metric]
                        row.append(fmt(m["mean"], m["std"]))
                lines.append("\t".join(row))
        lines.append(f"Fonte: Elaborado pelos autores ({ano}).")
        lines.append("")
        lines.append("")
    return "\n".join(lines)


def print_terminal(results: dict, rep: str) -> None:
    for n in dims_of(results):
        any_report = next(r for (d, _), r in results.items() if d == n)
        print(f"\n{'='*100}")
        print(f"  {rep_label(rep)}  |  N = {n_label(n, any_report)}")
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
            report = results.get((n, model))
            print(f"  {MODEL_LABELS[model]:<25}", end="")
            for metric in METRIC_LABELS:
                if report and metric in report["metrics"]:
                    m = report["metrics"][metric]
                    cell = fmt(m["mean"], m["std"])
                else:
                    cell = "N/D"
                print(f"  {cell:>{col}}", end="")
            print()


def write_tables(dataset: str, current_rep: str | None) -> None:
    """Refaz tabelas_resultados.tsv de cada representação e o resumo entre representações."""
    base_reports_dir = dataset_paths(dataset)["reports_dir"]
    all_results = {}
    for rep, reports_dir in rep_dirs(base_reports_dir).items():
        results = load_results(reports_dir)
        if not results:
            continue
        all_results[rep] = results
        with open(os.path.join(reports_dir, "tabelas_resultados.tsv"), "w", encoding="utf-8") as f:
            f.write(build_tsv(results, dataset, rep))
        if rep == current_rep:
            print_terminal(results, rep)

    summary_path = os.path.join(base_reports_dir, "tabelas_representacoes.tsv")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(build_summary_tsv(all_results, dataset))

    print(f"\n{'='*60}")
    print(f"  Tabelas por representação: {{relatórios}}/tabelas_resultados.tsv "
          f"({', '.join(all_results)})")
    print(f"  Resumo entre representações: {summary_path}")
    print(f"  Para usar no Google Docs:")
    print(f"    1. Abra o arquivo .tsv")
    print(f"    2. Selecione todo o conteúdo de uma tabela")
    print(f"    3. Cole no Google Docs — ele cria a tabela automaticamente")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    add_dataset_arg(parser)
    add_representation_args(parser, n_components=False)
    parser.add_argument("--n-values", type=int, nargs="+", default=N_VALUES,
                        help=f"Tamanhos de vetor; 0 = sem projeção (padrão: {N_VALUES}).")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS,
                        help="Modelos a rodar (padrão: todos).")
    parser.add_argument("--resumo", action="store_true",
                        help="Não roda nada; só refaz as tabelas a partir dos JSONs existentes.")
    args = parser.parse_args()
    patterns = patterns_from_args(args)
    rep = rep_name(patterns)
    os.makedirs(dataset_paths(args.dataset, patterns)["reports_dir"], exist_ok=True)

    if args.resumo:
        write_tables(args.dataset, rep)
        return

    common_args = ["--dataset", args.dataset] + (
        ["--pattern", *patterns] if args.pattern else ["--k", str(args.k)])

    print(f"\n{'#'*60}")
    print(f"# Matrizes k-mer — {rep_label(rep)}")
    print(f"{'#'*60}")
    if run([sys.executable, "src/01_kmer_matrix.py"] + common_args) != 0:
        print("[ERRO] 01_kmer_matrix.py falhou.")
        sys.exit(1)

    for n in args.n_values:
        n_args = common_args + ["--n-components", str(n)]
        label = f"N={n}" if n else "sem projeção"
        print(f"\n{'#'*60}")
        print(f"# Gerando vetores {rep_label(rep)}, {label}")
        print(f"{'#'*60}")
        rc = run([sys.executable, "src/02_project_random.py"] + n_args)
        if rc != 0:
            print(f"[ERRO] 02_project_random.py falhou para {label}.")
            sys.exit(1)

        for model in args.models:
            print(f"\n  → {model}  ({rep_label(rep)}, {label})")
            rc = run([sys.executable, "src/05_train_evaluate.py", "--model", model] + n_args)
            if rc != 0:
                print(f"  [ERRO] {model} falhou para {label}.")

    write_tables(args.dataset, rep)


if __name__ == "__main__":
    main()
