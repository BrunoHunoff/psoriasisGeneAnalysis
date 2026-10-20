"""
Módulo 08 — Teste cego do TCC II (validação externa prospectiva).

Avalia o(s) classificador(es) FINAL(is) em proteínas que não participaram do treino nem
da escolha de hiperparâmetros:
  - positivos: genes dos loci "Newly reported" de Dand et al. (2025) — data/teste_cego_v2/
  - negativos: genes sorteados à parte por 00_sample_negatives.py — data/teste_cego_negativos_v2/

⚠️ RODE UMA ÚNICA VEZ, depois de escolher o modelo final pela validação cruzada.
Se o resultado for usado para voltar e ajustar o modelo, o teste deixa de ser cego.
Por isso o script recusa sobrescrever um resultado existente sem --force.

A sequência é vetorizada com o MESMO projetor do treino (models/v2/random_projection.pkl).
Métricas em dois níveis:
  - por sequência (cada homólogo conta)
  - por gene (média de P(positivo) entre os homólogos do gene) — a unidade biológica
e quebradas por nível de evidência dos positivos (A/B/C/D, dand2025_validacao_ineditos.tsv)
e por nível de dificuldade dos negativos (fácil/intermediário/difícil).

Saídas (reports/v2/):
  teste_cego_{modelo}.json   — métricas e predições por sequência e por gene
  tabelas_teste_cego.tsv     — tabelas para o Google Docs (todos os modelos já avaliados)

Pré-requisito:
    python src/run_experiments.py --dataset tcc2
    python src/06_train_final.py --model <modelo> --dataset tcc2

Uso:
    python src/08_blind_test.py --model logistic_regression
    python src/08_blind_test.py --model all
"""

import os
import sys
import csv
import json
import pickle
import logging
import argparse
from datetime import date, datetime

import numpy as np
from Bio import SeqIO
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

from common import (FASTA_EXTS, build_vocab, classifier_path, dataset_paths, projector_path,
                    sequence_to_vector)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DATASET         = "tcc2"   # único dataset com conjunto de teste cego
EVIDENCE_TSV    = "data/gene_lists/dand2025_validacao_ineditos.tsv"
THRESHOLD       = 0.5
MODELS = {
    "logistic_regression": "Regressão Logística",
    "svm":                 "SVM",
    "random_forest":       "Random Forest",
    "gradient_boosting":   "Gradient Boosting",
    "naive_bayes":         "Naive Bayes",
    "knn":                 "kNN",
}
METRIC_LABELS = {
    "accuracy":    "Acurácia",
    "precision":   "Precisão",
    "sensitivity": "Sensibilidade",
    "specificity": "Especificidade",
    "f1":          "F1 Score",
    "mcc":         "MCC",
    "roc_auc":     "ROC-AUC",
}
EVIDENCE_LEVELS = {
    "A": "A — variante codificante / alvo terapêutico",
    "B": "B — TWAS",
    "C": "C — sinal dentro do gene",
    "D": "D — gene mais próximo (intergênico)",
}
NEG_LEVELS = {
    "facil":         "Fácil (housekeeping)",
    "intermediario": "Intermediário (outras doenças imunes)",
    "dificil":       "Difícil (imunes não-psoríase e parálogos)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teste cego do TCC II (rodar uma única vez).")
    parser.add_argument("--model", required=True, choices=list(MODELS) + ["all"],
                        help="Modelo final a avaliar, ou 'all' para todos os treinados.")
    parser.add_argument("--force", action="store_true",
                        help="Sobrescreve um resultado de teste cego já existente.")
    return parser.parse_args()


def load_evidence_levels(path: str) -> dict[str, str]:
    """gene → nível de evidência (A/B/C/D) das regiões Newly reported usadas no teste."""
    levels = {}
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["excluir_validacao"] != "nao":
                continue
            for gene in row["genes_escolhidos"].split(","):
                if gene.strip():
                    levels[gene.strip().upper()] = row["nivel_evidencia"]
    return levels


def load_blind_set(pos_dir: str, neg_dir: str, evidence: dict[str, str]) -> list[dict]:
    """Uma entrada por sequência: gene, label, nível (evidência ou dificuldade), seq."""
    records = []
    for directory, label in ((pos_dir, 1), (neg_dir, 0)):
        if not os.path.isdir(directory):
            log.error("%s não encontrado — rode a coleta (00_fetch_uniprot.py / 00_sample_negatives.py).",
                      directory)
            sys.exit(1)
        for fname in sorted(os.listdir(directory)):
            if os.path.splitext(fname)[1].lower() not in FASTA_EXTS:
                continue
            stem = os.path.splitext(fname)[0]
            if label == 1:
                gene = stem.upper()
                level = evidence.get(gene, "?")
            else:
                gene, level = stem.split("_")[0].upper(), stem.split("_")[1]
            for rec in SeqIO.parse(os.path.join(directory, fname), "fasta"):
                records.append({"gene": gene, "label": label, "nivel": level,
                                "seq_id": rec.id, "seq": str(rec.seq).upper()})
    return records


def compute_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    y_pred = (proba >= THRESHOLD).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return {
        "n":           int(len(y_true)),
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "precision":   float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc":         float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc":     float(roc_auc_score(y_true, proba)),
    }


def gene_table(records: list[dict], proba: np.ndarray) -> list[dict]:
    """Agrega por gene: média de P(positivo) entre os homólogos."""
    genes: dict[tuple, list[float]] = {}
    for rec, p in zip(records, proba):
        genes.setdefault((rec["gene"], rec["label"], rec["nivel"]), []).append(float(p))
    return [{"gene": g, "label": lbl, "nivel": lvl, "n_seqs": len(ps),
             "proba_media": float(np.mean(ps)), "pred": int(np.mean(ps) >= THRESHOLD)}
            for (g, lbl, lvl), ps in sorted(genes.items())]


def breakdown(genes: list[dict], label: int) -> dict:
    """Taxa de acerto por nível — sensibilidade (positivos) ou especificidade (negativos)."""
    out = {}
    for g in genes:
        if g["label"] != label:
            continue
        d = out.setdefault(g["nivel"], {"n_genes": 0, "acertos": 0})
        d["n_genes"] += 1
        d["acertos"] += int(g["pred"] == label)
    for d in out.values():
        d["taxa"] = d["acertos"] / d["n_genes"]
    return dict(sorted(out.items()))


def pct(x: float) -> str:
    return f"{x*100:.1f}%".replace(".", ",")


def build_tsv(results: dict) -> str:
    ano = date.today().year
    fonte = f"Fonte: Elaborado pelos autores ({ano})."
    lines = []
    for unit, title in (("por_sequencia", "por sequência"), ("por_gene", "por gene (média entre homólogos)")):
        n = next(iter(results.values()))["metricas"][unit]["n"]
        lines += [f"Tabela — Teste cego (loci inéditos de Dand et al., 2025), métricas {title}; n={n}", "",
                  "\t".join(["Modelo"] + list(METRIC_LABELS.values()))]
        for model, r in results.items():
            m = r["metricas"][unit]
            lines.append("\t".join([MODELS[model]] + [pct(m[k]) if k != "mcc" else f"{m[k]:.3f}".replace(".", ",")
                                                     for k in METRIC_LABELS]))
        lines += [fonte, "", ""]

    lines += ["Tabela — Teste cego: sensibilidade por nível de evidência do locus (por gene)", "",
              "\t".join(["Modelo"] + list(EVIDENCE_LEVELS.values()))]
    for model, r in results.items():
        row = [MODELS[model]]
        for lvl in EVIDENCE_LEVELS:
            d = r["sensibilidade_por_evidencia"].get(lvl)
            row.append(f"{d['acertos']}/{d['n_genes']} ({pct(d['taxa'])})" if d else "N/D")
        lines.append("\t".join(row))
    lines += [fonte, "", ""]

    lines += ["Tabela — Teste cego: especificidade por nível de dificuldade do negativo (por gene)", "",
              "\t".join(["Modelo"] + list(NEG_LEVELS.values()))]
    for model, r in results.items():
        row = [MODELS[model]]
        for lvl in NEG_LEVELS:
            d = r["especificidade_por_nivel"].get(lvl)
            row.append(f"{d['acertos']}/{d['n_genes']} ({pct(d['taxa'])})" if d else "N/D")
        lines.append("\t".join(row))
    lines += [fonte]
    return "\n".join(lines)


def main():
    args = parse_args()
    paths = dataset_paths(DATASET)
    models_dir, reports_dir = paths["models_dir"], paths["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)

    proj_path = projector_path(models_dir)
    if not os.path.exists(proj_path):
        log.error("%s não encontrado — rode run_experiments.py --dataset %s e 06_train_final.py antes.",
                  proj_path, DATASET)
        sys.exit(1)
    with open(proj_path, "rb") as f:
        projector = pickle.load(f)
    n_components = projector.n_components_

    models = list(MODELS) if args.model == "all" else [args.model]
    classifiers, metas = {}, {}
    for model in models:
        clf_path = classifier_path(models_dir, model, n_components)
        meta_path = clf_path.replace(".pkl", "_meta.json")
        if not os.path.exists(clf_path) or not os.path.exists(meta_path):
            msg = (f"Modelo final não encontrado: {clf_path} (+ _meta.json) — rode "
                   f"06_train_final.py --model {model} --dataset {DATASET}")
            if args.model == "all":
                log.warning(msg + ". Pulando.")
                continue
            log.error(msg)
            sys.exit(1)
        out_json = os.path.join(reports_dir, f"teste_cego_{model}.json")
        if os.path.exists(out_json) and not args.force:
            log.error("O teste cego de %s já foi rodado (%s). Rodar de novo só faz sentido se o modelo "
                      "NÃO foi alterado com base nesse resultado; nesse caso use --force.", model, out_json)
            sys.exit(1)
        with open(clf_path, "rb") as f:
            classifiers[model] = pickle.load(f)
        with open(meta_path) as f:
            metas[model] = json.load(f)
    if not classifiers:
        log.error("Nenhum modelo final disponível para N=%d.", n_components)
        sys.exit(1)

    evidence = load_evidence_levels(EVIDENCE_TSV)
    records = load_blind_set(paths["blind_pos_dir"], paths["blind_neg_dir"], evidence)
    _, vocab_index = build_vocab()
    vectors, kept = [], []
    for rec in records:
        v = sequence_to_vector(rec["seq"], vocab_index, projector)
        if v is None:
            log.warning("Sequência curta demais, ignorada: %s", rec["seq_id"])
            continue
        vectors.append(v)
        kept.append(rec)
    X = np.vstack(vectors)
    y = np.array([r["label"] for r in kept])
    log.info("Teste cego: %d sequências (pos=%d, neg=%d), N=%d",
             len(y), int(y.sum()), int((y == 0).sum()), n_components)

    results = {}
    for model, clf in classifiers.items():
        proba = clf.predict_proba(X)[:, 1]
        genes = gene_table(kept, proba)
        g_true = np.array([g["label"] for g in genes])
        g_proba = np.array([g["proba_media"] for g in genes])
        result = {
            "modelo": model,
            "dataset": DATASET,
            "n_components": n_components,
            "hiperparametros": metas[model].get("best_params"),
            "limiar": THRESHOLD,
            "executado_em": datetime.now().isoformat(timespec="seconds"),
            "metricas": {
                "por_sequencia": compute_metrics(y, proba),
                "por_gene": compute_metrics(g_true, g_proba),
            },
            "sensibilidade_por_evidencia": breakdown(genes, 1),
            "especificidade_por_nivel": breakdown(genes, 0),
            "genes": genes,
            "sequencias": [{"seq_id": r["seq_id"], "gene": r["gene"], "label": r["label"],
                            "nivel": r["nivel"], "proba": float(p)} for r, p in zip(kept, proba)],
        }
        with open(os.path.join(reports_dir, f"teste_cego_{model}.json"), "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        results[model] = result
        ms, mg = result["metricas"]["por_sequencia"], result["metricas"]["por_gene"]
        log.info("%-20s  seq: ROC-AUC=%.3f sens=%.3f esp=%.3f | gene: ROC-AUC=%.3f sens=%.3f esp=%.3f",
                 model, ms["roc_auc"], ms["sensitivity"], ms["specificity"],
                 mg["roc_auc"], mg["sensitivity"], mg["specificity"])

    # A tabela reúne todos os modelos já avaliados, não só os desta execução
    all_results = {}
    for model in MODELS:
        path = os.path.join(reports_dir, f"teste_cego_{model}.json")
        if os.path.exists(path):
            with open(path) as f:
                all_results[model] = json.load(f)
    tsv_path = os.path.join(reports_dir, "tabelas_teste_cego.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write(build_tsv(all_results))
    log.info("Resultados: %s/teste_cego_{modelo}.json  |  Tabelas: %s", reports_dir, tsv_path)


if __name__ == "__main__":
    main()
