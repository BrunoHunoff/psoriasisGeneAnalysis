"""
Módulo 08 — Teste cego do TCC II (validação externa prospectiva).

Avalia o(s) classificador(es) FINAL(is) em proteínas que não participaram do treino nem
da escolha de hiperparâmetros:
  - positivos: genes dos loci "Newly reported" de Dand et al. (2025) — data/teste_cego_v2/
  - negativos: genes sorteados à parte por 00_sample_negatives.py — data/teste_cego_negativos_v2/

⚠️ RODE UMA ÚNICA VEZ POR BRAÇO (--tag), seguindo docs/plano_otimizacao.md:
  antes                — braço 0: modelos da linha de base (grids do 05)
  otimizado_misturado  — braço A: otimização de hiperparâmetros com CV de genes misturados
  otimizado_por_gene   — braço B: a mesma otimização com CV separada por gene
O plano (espaços de busca, orçamento, métrica, regra do modelo principal) foi registrado ANTES
do braço 0; se o resultado de um braço for usado para voltar e ajustar os modelos, o teste
deixa de ser cego. Por isso o script recusa sobrescrever um resultado existente sem --force,
e --n-components é obrigatório (o padrão 1000 dos outros scripts não vale aqui).

A sequência é vetorizada com a mesma representação e a mesma projeção do treino (refeita a
partir de representação, N e seed — common.kmer_vectors); a representação do _meta.json do
classificador precisa bater com a pedida.
Métricas em dois níveis:
  - por sequência (cada homólogo conta)
  - por gene (média de P(positivo) entre os homólogos do gene) — a unidade biológica
e quebradas por nível de evidência dos positivos (A/B/C/D, dand2025_validacao_ineditos.tsv)
e por nível de dificuldade dos negativos (fácil/intermediário/difícil).

Braços A e B: modelos de models/v2/otimizacao/{misturado|por_gene}/ (09_optimize_hyperparams.py) —
LR, SVM, árvores, boosting, MLP e comitê (média das probabilidades dos cinco). Cada um usa o limiar
de decisão ajustado na validação do braço (_meta.json); as métricas saem nesse limiar (principal)
e também com 0,5, para comparar com o braço 0.

Saídas (reports/v2/, qualquer que seja a representação — ela fica registrada no JSON):
  teste_cego_{tag}_{modelo}.json   — métricas e predições por sequência e por gene
  tabelas_teste_cego_{tag}.tsv     — tabelas para o Google Docs (todos os modelos já avaliados na etapa)

Pré-requisito (mesma representação e mesmo --n-components):
    braço 0:     python src/06_train_final.py --model <modelo> --dataset tcc2 --n-components 4000
    braços A/B:  python src/09_optimize_hyperparams.py --arm {misturado|por_gene} --model all --n-components 4000

Uso:
    python src/08_blind_test.py --tag antes --model all --n-components 4000
    python src/08_blind_test.py --tag otimizado_por_gene --model all --n-components 4000
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

from common import (FASTA_EXTS, add_representation_args, classifier_path, dataset_paths,
                    patterns_from_args, rep_name, sequences_to_vectors, vector_dim)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DATASET         = "tcc2"   # único dataset com conjunto de teste cego
EVIDENCE_TSV    = "data/gene_lists/dand2025_validacao_ineditos.tsv"
THRESHOLD       = 0.5
TAGS            = ("antes", "otimizado_misturado", "otimizado_por_gene")
TAG_LABELS = {
    "antes":               "braço 0 — linha de base",
    "otimizado_misturado": "braço A — otimizado, CV de genes misturados",
    "otimizado_por_gene":  "braço B — otimizado, CV separada por gene",
}
TAG_ARM = {"otimizado_misturado": "misturado", "otimizado_por_gene": "por_gene"}   # tag → pasta do 09
OPT_MODELS = {
    "logistic_regression": "Regressão Logística",
    "svm":                 "SVM",
    "trees":               "Árvores (RF/ET)",
    "boosting":            "Boosting (GB/HistGB)",
    "mlp":                 "MLP",
    "comite":              "Comitê",
}
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
    parser = argparse.ArgumentParser(description="Teste cego do TCC II (rodar uma única vez por etapa).")
    parser.add_argument("--model", required=True, choices=list(dict.fromkeys([*MODELS, *OPT_MODELS])) + ["all"],
                        help="Modelo final a avaliar, ou 'all' para todos os do braço.")
    parser.add_argument("--tag", required=True, choices=TAGS,
                        help="Braço do experimento (docs/plano_otimizacao.md).")
    parser.add_argument("--force", action="store_true",
                        help="Sobrescreve um resultado de teste cego já existente.")
    add_representation_args(parser, require_n_components=True)
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


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = THRESHOLD) -> dict:
    y_pred = (proba >= threshold).astype(int)
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


def gene_table(records: list[dict], proba: np.ndarray, threshold: float = THRESHOLD) -> list[dict]:
    """Agrega por gene: média de P(positivo) entre os homólogos."""
    genes: dict[tuple, list[float]] = {}
    for rec, p in zip(records, proba):
        genes.setdefault((rec["gene"], rec["label"], rec["nivel"]), []).append(float(p))
    return [{"gene": g, "label": lbl, "nivel": lvl, "n_seqs": len(ps),
             "proba_media": float(np.mean(ps)), "pred": int(np.mean(ps) >= threshold)}
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


def config_label(r: dict) -> str:
    n = f"N={r['n_components']}" if r.get("projecao", True) else f"sem projeção ({r['n_components']})"
    return f"{r.get('representacao', 'k3')}, {n}"


def build_tsv(results: dict, tag: str) -> str:
    ano = date.today().year
    fonte = f"Fonte: Elaborado pelos autores ({ano})."
    configs = {config_label(r) for r in results.values()}
    # Uma representação para todos os modelos → vai no título; senão, ao lado de cada modelo
    suffix = f"; {configs.pop()}" if len(configs) == 1 else ""
    names = MODELS if tag == "antes" else OPT_MODELS
    label = {m: names[m] + ("" if suffix else f" ({config_label(r)})")
             + ("" if tag == "antes" else f" — limiar {r['limiar']:.2f}".replace(".", ","))
             for m, r in results.items()}
    lines = []
    tables = [("metricas", "", label)]
    if tag != "antes":   # braços A e B: limiar ajustado (principal) e 0,5 (comparação com o braço 0)
        tables = [("metricas", ", limiar ajustado na validação", label),
                  ("metricas_limiar_0_5", ", limiar 0,5", {m: names[m] for m in results})]
    for key, thr_title, row_label in tables:
        for unit, title in (("por_sequencia", "por sequência"), ("por_gene", "por gene (média entre homólogos)")):
            n = next(iter(results.values()))[key][unit]["n"]
            lines += [f"Tabela — Teste cego, {TAG_LABELS[tag]} (loci inéditos de Dand et al., 2025), "
                      f"métricas {title}{thr_title}; n={n}{suffix}", "",
                      "\t".join(["Modelo"] + list(METRIC_LABELS.values()))]
            for model, r in results.items():
                m = r[key][unit]
                lines.append("\t".join([row_label[model]] + [pct(m[k]) if k != "mcc" else f"{m[k]:.3f}".replace(".", ",")
                                                            for k in METRIC_LABELS]))
            lines += [fonte, "", ""]

    lines += [f"Tabela — Teste cego, {TAG_LABELS[tag]}: sensibilidade por nível de evidência do locus "
              f"(por gene){suffix}", "",
              "\t".join(["Modelo"] + list(EVIDENCE_LEVELS.values()))]
    for model, r in results.items():
        row = [label[model]]
        for lvl in EVIDENCE_LEVELS:
            d = r["sensibilidade_por_evidencia"].get(lvl)
            row.append(f"{d['acertos']}/{d['n_genes']} ({pct(d['taxa'])})" if d else "N/D")
        lines.append("\t".join(row))
    lines += [fonte, "", ""]

    lines += [f"Tabela — Teste cego, {TAG_LABELS[tag]}: especificidade por nível de dificuldade do "
              f"negativo (por gene){suffix}", "",
              "\t".join(["Modelo"] + list(NEG_LEVELS.values()))]
    for model, r in results.items():
        row = [label[model]]
        for lvl in NEG_LEVELS:
            d = r["especificidade_por_nivel"].get(lvl)
            row.append(f"{d['acertos']}/{d['n_genes']} ({pct(d['taxa'])})" if d else "N/D")
        lines.append("\t".join(row))
    lines += [fonte]
    return "\n".join(lines)


def main():
    args = parse_args()
    patterns = patterns_from_args(args)
    rep = rep_name(patterns)
    n_components = vector_dim(patterns, args.n_components)
    paths = dataset_paths(DATASET, patterns)
    optimized = args.tag != "antes"
    names = OPT_MODELS if optimized else MODELS
    models_dir = (os.path.join(paths["models_dir"], "otimizacao", TAG_ARM[args.tag]) if optimized
                  else paths["models_dir"])
    reports_dir = dataset_paths(DATASET)["reports_dir"]   # reports/v2/, para qualquer representação
    os.makedirs(reports_dir, exist_ok=True)
    if args.model != "all" and args.model not in names:
        log.error("O modelo %s não faz parte do %s (modelos: %s).", args.model, TAG_LABELS[args.tag], ", ".join(names))
        sys.exit(1)
    prereq = ("06_train_final.py --model {m} --dataset tcc2" if not optimized else
              "09_optimize_hyperparams.py --arm " + TAG_ARM.get(args.tag, "") + " --model {m}")

    models = list(names) if args.model == "all" else [args.model]
    classifiers, metas = {}, {}
    for model in models:
        clf_path = classifier_path(models_dir, model, n_components)
        meta_path = clf_path.replace(".pkl", "_meta.json")
        needed = ([classifier_path(models_dir, m, n_components) for m in json.load(open(meta_path))["membros"]]
                  if model == "comite" and os.path.exists(meta_path) else [clf_path])
        if not os.path.exists(meta_path) or not all(os.path.exists(p) for p in needed):
            msg = (f"Modelo final não encontrado: {clf_path} (+ _meta.json) — rode "
                   f"{prereq.format(m=model)} com a mesma representação e o mesmo --n-components")
            if args.model == "all":
                log.warning(msg + ". Pulando.")
                continue
            log.error(msg)
            sys.exit(1)
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("patterns", ["111"]) != list(patterns) or meta.get("n_components") != n_components:
            log.error("%s foi treinado com %s, N=%s — não bate com %s, N=%d. Rode 06_train_final.py de novo.",
                      meta_path, meta.get("representation", "k3"), meta.get("n_components"), rep, n_components)
            sys.exit(1)
        out_json = os.path.join(reports_dir, f"teste_cego_{args.tag}_{model}.json")
        if os.path.exists(out_json) and not args.force:
            log.error("O teste cego %s de %s já foi rodado (%s). Rodar de novo só faz sentido se o modelo "
                      "NÃO foi alterado com base nesse resultado; nesse caso use --force.",
                      args.tag, model, out_json)
            sys.exit(1)
        loaded = []
        for path in needed:
            with open(path, "rb") as f:
                loaded.append(pickle.load(f))
        classifiers[model] = loaded   # um modelo, ou os membros do comitê
        metas[model] = meta
    if not classifiers:
        log.error("Nenhum modelo final disponível para %s, N=%d.", rep, n_components)
        sys.exit(1)

    evidence = load_evidence_levels(EVIDENCE_TSV)
    records = load_blind_set(paths["blind_pos_dir"], paths["blind_neg_dir"], evidence)
    X, kept_idx = sequences_to_vectors([r["seq"] for r in records], patterns, args.n_components)
    for i in sorted(set(range(len(records))) - set(kept_idx)):
        log.warning("Sequência curta demais, ignorada: %s", records[i]["seq_id"])
    kept = [records[i] for i in kept_idx]
    y = np.array([r["label"] for r in kept])
    log.info("Teste cego %s: %d sequências (pos=%d, neg=%d), representação %s, N=%d",
             args.tag, len(y), int(y.sum()), int((y == 0).sum()), rep, n_components)

    results = {}
    for model, clfs in classifiers.items():
        proba = np.mean([c.predict_proba(X)[:, 1] for c in clfs], axis=0)   # comitê: média dos membros
        threshold = metas[model].get("limiar", THRESHOLD) if optimized else THRESHOLD
        genes = gene_table(kept, proba, threshold)
        g_true = np.array([g["label"] for g in genes])
        g_proba = np.array([g["proba_media"] for g in genes])
        result = {
            "modelo": model,
            "dataset": DATASET,
            "etapa": args.tag,
            "representacao": rep,
            "padroes": list(patterns),
            "projecao": bool(args.n_components),
            "n_components": n_components,
            "hiperparametros": metas[model].get("best_params", {"membros": metas[model].get("membros")}),
            "limiar": threshold,
            "executado_em": datetime.now().isoformat(timespec="seconds"),
            "metricas": {
                "por_sequencia": compute_metrics(y, proba, threshold),
                "por_gene": compute_metrics(g_true, g_proba, threshold),
            },
            "sensibilidade_por_evidencia": breakdown(genes, 1),
            "especificidade_por_nivel": breakdown(genes, 0),
            "genes": genes,
            "sequencias": [{"seq_id": r["seq_id"], "gene": r["gene"], "label": r["label"],
                            "nivel": r["nivel"], "proba": float(p)} for r, p in zip(kept, proba)],
        }
        if optimized:
            result["metricas_limiar_0_5"] = {"por_sequencia": compute_metrics(y, proba),
                                             "por_gene": compute_metrics(g_true, g_proba)}
        with open(os.path.join(reports_dir, f"teste_cego_{args.tag}_{model}.json"), "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        results[model] = result
        ms, mg = result["metricas"]["por_sequencia"], result["metricas"]["por_gene"]
        log.info("%-20s  limiar %.2f  seq: ROC-AUC=%.3f sens=%.3f esp=%.3f | gene: ROC-AUC=%.3f MCC=%.3f "
                 "sens=%.3f esp=%.3f", model, threshold, ms["roc_auc"], ms["sensitivity"], ms["specificity"],
                 mg["roc_auc"], mg["mcc"], mg["sensitivity"], mg["specificity"])

    # A tabela reúne todos os modelos já avaliados na etapa, não só os desta execução
    all_results = {}
    for model in names:
        path = os.path.join(reports_dir, f"teste_cego_{args.tag}_{model}.json")
        if os.path.exists(path):
            with open(path) as f:
                all_results[model] = json.load(f)
    tsv_path = os.path.join(reports_dir, f"tabelas_teste_cego_{args.tag}.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write(build_tsv(all_results, args.tag))
    log.info("Resultados: %s/teste_cego_%s_{modelo}.json  |  Tabelas: %s", reports_dir, args.tag, tsv_path)


if __name__ == "__main__":
    main()
