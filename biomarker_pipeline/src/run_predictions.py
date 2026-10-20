"""
Script de predição em lote para proteínas de validação.

Roda todos os modelos treinados nas proteínas de data/input/ (validação externa do
TCC I) e gera tabela no formato de validação em {reports_dir}/tabelas_predicoes.tsv.
No TCC II as mesmas 5 proteínas são repetidas para comparar com o TCC I.

Representação (--k / --pattern) e N (--n-components, 0 = sem projeção) escolhem os
classificadores; a tabela vai para o diretório de relatórios da representação.

Uso:
    python src/run_predictions.py --dataset tcc2
    python src/run_predictions.py --dataset tcc2 --k 4 --n-components 4000
"""

import os
import sys
import pickle
import argparse
from datetime import date
from Bio import SeqIO

from common import (INPUT_DIR, add_dataset_arg, add_representation_args, classifier_path,
                    dataset_paths, patterns_from_args, rep_name, sequences_to_vectors, vector_dim)

MODELS = [
    ("logistic_regression", "LR"),
    ("svm",                 "SVM"),
    ("random_forest",       "RF"),
    ("gradient_boosting",   "GB"),
    ("naive_bayes",         "NB"),
    ("knn",                 "kNN"),
]

# Metadados de cada proteína de validação, por dataset: nome_arquivo → (nome, categoria, esperado)
# esperado: 1 = POSITIVO, 0 = NEGATIVO
PROTEIN_META = {
    # Rótulos como publicados no TCC I (a PTPN22 humana estava, na verdade, nos negativos de treino)
    "tcc1": {
        "INS_housekeeping_negativa_treinada":    ("INS",    "Negativa treinada",                    0),
        "PTPN22_Artrite_Reumatoide":             ("PTPN22", "Negativa não treinada",                0),
        "IL12B_Postivia_Treinada":               ("IL12B",  "Positiva treinada",                    1),
        "CARD14_Postiva":                        ("CARD14", "Positiva não treinada — epitelial",     1),
        "RUNX3_Positiva":                        ("RUNX3",  "Positiva não treinada — imune sistêmica", 1),
    },
    # TCC II: INS e PTPN22 foram excluídas dos negativos pelos critérios v2; RUNX3 e CARD14
    # reservadas fora do treino (00_build_gene_lists.TCC1_VALIDATION_HOLDOUT)
    "tcc2": {
        "INS_housekeeping_negativa_treinada":    ("INS",    "Negativa não treinada (excluída dos negativos v2)", 0),
        "PTPN22_Artrite_Reumatoide":             ("PTPN22", "Negativa não treinada",                0),
        "IL12B_Postivia_Treinada":               ("IL12B",  "Positiva treinada",                    1),
        "CARD14_Postiva":                        ("CARD14", "Positiva não treinada — epitelial",     1),
        "RUNX3_Positiva":                        ("RUNX3",  "Positiva não treinada — imune sistêmica", 1),
    },
}

CONFIDENCE_THRESHOLD = 65.0  # abaixo disso → ~


def fmt_result(pred: int, confidence: float, expected: int) -> str:
    correct = (pred == expected)
    if not correct:
        return f"✗ {confidence:.1f}%".replace(".", ",")
    if confidence < CONFIDENCE_THRESHOLD:
        return f"~ {confidence:.1f}%".replace(".", ",")
    return f"✓ {confidence:.1f}%".replace(".", ",")


def main():
    parser = argparse.ArgumentParser()
    add_dataset_arg(parser)
    add_representation_args(parser)
    args = parser.parse_args()
    patterns = patterns_from_args(args)
    n_components = vector_dim(patterns, args.n_components)
    rep = rep_name(patterns)
    paths = dataset_paths(args.dataset, patterns)
    models_dir, reports_dir = paths["models_dir"], paths["reports_dir"]
    tsv_path = os.path.join(reports_dir, "tabelas_predicoes.tsv")
    os.makedirs(reports_dir, exist_ok=True)
    print(f"Dataset {args.dataset} — representação {rep}, N={n_components}"
          f"{'' if args.n_components else ' (sem projeção)'}")

    # carrega sequências
    sequences = []
    for stem, meta in PROTEIN_META[args.dataset].items():
        found = False
        for ext in (".fasta", ".fa"):
            path = os.path.join(INPUT_DIR, stem + ext)
            if os.path.exists(path):
                recs = list(SeqIO.parse(path, "fasta"))
                if recs:
                    sequences.append((stem, str(recs[0].seq).upper(), meta))
                    found = True
                    break
        if not found:
            print(f"[AVISO] Arquivo não encontrado: {stem}.fasta — pulando.")

    if not sequences:
        print("[ERRO] Nenhuma sequência carregada.")
        sys.exit(1)

    # carrega classificadores
    classifiers = {}
    for model_key, _ in MODELS:
        clf_path = classifier_path(models_dir, model_key, n_components)
        if os.path.exists(clf_path):
            with open(clf_path, "rb") as f:
                classifiers[model_key] = pickle.load(f)
        else:
            print(f"[AVISO] Modelo não encontrado: {clf_path}")

    if not classifiers:
        print(f"[ERRO] Nenhum classificador para N={n_components} em {models_dir} — rode "
              f"06_train_final.py com a mesma representação e o mesmo --n-components.")
        sys.exit(1)

    # predições
    results = {}
    X_all, kept = sequences_to_vectors([seq for _, seq, _ in sequences], patterns, args.n_components)
    for row, idx in enumerate(kept):
        stem = sequences[idx][0]
        X = X_all[row].reshape(1, -1)
        for model_key, _ in MODELS:
            clf = classifiers.get(model_key)
            if clf is None:
                continue
            pred  = int(clf.predict(X)[0])
            proba = clf.predict_proba(X)[0]
            conf  = float(proba[pred] * 100)
            results[(stem, model_key)] = (pred, conf)

    # ── imprime tabela no terminal ──────────────────────────────────────────
    col_w = 14
    header = f"  {'Proteína':<10}  {'Categoria':<42}  {'Esperado':<10}"
    for _, short in MODELS:
        header += f"  {short:>{col_w}}"
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"  {'-'*10}  {'-'*42}  {'-'*10}" + f"  {'─'*col_w}" * len(MODELS))

    for stem, _, (name, categoria, expected) in sequences:
        esperado_str = "POSITIVO" if expected == 1 else "NEGATIVO"
        row = f"  {name:<10}  {categoria:<42}  {esperado_str:<10}"
        for model_key, _ in MODELS:
            res = results.get((stem, model_key))
            if res is None:
                cell = "N/D"
            else:
                pred, conf = res
                cell = fmt_result(pred, conf, expected)
            row += f"  {cell:>{col_w}}"
        print(row)

    print(f"{'='*len(header)}\n")
    print(f"✓ = predição correta  ✗ = predição incorreta  ~ = correta mas confiança < {CONFIDENCE_THRESHOLD:.0f}%\n")

    # ── gera TSV para Google Docs ───────────────────────────────────────────
    ano = date.today().year
    lines = []
    lines.append(
        f"Tabela — Validação dos classificadores em proteínas de teste "
        f"(dataset {args.dataset}, representação {rep}, N={n_components}"
        f"{'' if args.n_components else ' sem projeção'}, modelos treinados com GridSearchCV 10-fold CV)"
    )
    lines.append("")

    tsv_header = ["Proteína", "Categoria", "Classif. esperada"] + [short for _, short in MODELS]
    lines.append("\t".join(tsv_header))

    for stem, _, (name, categoria, expected) in sequences:
        esperado_str = "POSITIVO" if expected == 1 else "NEGATIVO"
        row = [name, categoria, esperado_str]
        for model_key, _ in MODELS:
            res = results.get((stem, model_key))
            if res is None:
                row.append("N/D")
            else:
                pred, conf = res
                row.append(fmt_result(pred, conf, expected))
        lines.append("\t".join(row))

    lines.append(f"Fonte: Elaborado pelos autores ({ano}).")
    lines.append(f"✓ = predição correta  |  ✗ = predição incorreta  |  ~ = correta mas confiança < {CONFIDENCE_THRESHOLD:.0f}%")

    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Tabela salva em: {tsv_path}")
    print("Cole no Google Docs para formatar automaticamente.")


if __name__ == "__main__":
    main()
