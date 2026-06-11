"""
Script de predição em lote para proteínas de validação.

Roda todos os modelos treinados nas proteínas de data/input/ e gera
tabela no formato de validação em reports/tabelas_predicoes.tsv.

Uso:
    python src/run_predictions.py
"""

import os
import sys
import pickle
import itertools
import numpy as np
from datetime import date
from Bio import SeqIO

INPUT_DIR   = "data/input"
MODELS_DIR  = "models"
REPORTS_DIR = "reports"
TSV_PATH    = os.path.join(REPORTS_DIR, "tabelas_predicoes.tsv")

K            = 3
AMINO        = "ACDEFGHIKLMNPQRSTVWY"
N_COMPONENTS = 1000

MODELS = [
    ("logistic_regression", "LR"),
    ("svm",                 "SVM"),
    ("random_forest",       "RF"),
    ("gradient_boosting",   "GB"),
    ("naive_bayes",         "NB"),
    ("knn",                 "kNN"),
]

# Metadados de cada proteína de validação: nome_arquivo → (nome, categoria, esperado)
# esperado: 1 = POSITIVO, 0 = NEGATIVO
PROTEIN_META = {
    "INS_housekeeping_negativa_treinada":    ("INS",    "Negativa treinada",                    0),
    "PTPN22_Artrite_Reumatoide":             ("PTPN22", "Negativa não treinada",                0),
    "IL12B_Postivia_Treinada":               ("IL12B",  "Positiva treinada",                    1),
    "CARD14_Postiva":                        ("CARD14", "Positiva não treinada — epitelial",     1),
    "RUNX3_Positiva":                        ("RUNX3",  "Positiva não treinada — imune sistêmica", 1),
}

CONFIDENCE_THRESHOLD = 65.0  # abaixo disso → ~


def build_vocab() -> dict:
    vocab = ["".join(c) for c in itertools.product(AMINO, repeat=K)]
    return {kmer: idx for idx, kmer in enumerate(vocab)}


def sequence_to_vector(seq: str, vocab: dict, projector) -> np.ndarray | None:
    n_windows = len(seq) - K + 1
    if n_windows <= 0:
        return None
    n_vocab = len(vocab)
    matrix = np.zeros((n_windows, n_vocab), dtype=np.float32)
    for i in range(n_windows):
        kmer = seq[i: i + K]
        if kmer in vocab:
            matrix[i, vocab[kmer]] = 1.0
    return projector.transform(matrix).mean(axis=0).astype(np.float32)


def fmt_result(pred: int, confidence: float, expected: int) -> str:
    correct = (pred == expected)
    if not correct:
        return f"✗ {confidence:.1f}%".replace(".", ",")
    if confidence < CONFIDENCE_THRESHOLD:
        return f"~ {confidence:.1f}%".replace(".", ",")
    return f"✓ {confidence:.1f}%".replace(".", ",")


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    proj_path = os.path.join(MODELS_DIR, "random_projection.pkl")
    if not os.path.exists(proj_path):
        print(f"[ERRO] {proj_path} não encontrado. Rode 02_project_random.py primeiro.")
        sys.exit(1)
    with open(proj_path, "rb") as f:
        projector = pickle.load(f)
    print(f"Projector carregado: N={projector.n_components_} componentes")

    vocab = build_vocab()

    # carrega sequências
    sequences = []
    for stem, meta in PROTEIN_META.items():
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
        clf_path = os.path.join(MODELS_DIR, f"classifier_{model_key}_n{N_COMPONENTS}.pkl")
        if os.path.exists(clf_path):
            with open(clf_path, "rb") as f:
                classifiers[model_key] = pickle.load(f)
        else:
            print(f"[AVISO] Modelo não encontrado: {clf_path}")

    # predições
    results = {}
    for stem, seq, _ in sequences:
        vector = sequence_to_vector(seq, vocab, projector)
        if vector is None:
            continue
        X = vector.reshape(1, -1)
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
        f"(N={N_COMPONENTS} componentes, modelos treinados com GridSearchCV 10-fold CV)"
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

    with open(TSV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Tabela salva em: {TSV_PATH}")
    print("Cole no Google Docs para formatar automaticamente.")


if __name__ == "__main__":
    main()
