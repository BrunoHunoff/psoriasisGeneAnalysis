"""
Relatório — comparação final dos três braços do experimento do TCC II (docs/plano_otimizacao.md).

Para cada braço e modelo, junta a validação cruzada (do próprio braço) e o teste cego por gene,
com intervalo de confiança de 95% do ROC-AUC por bootstrap de genes (estratificado por classe).
Marca o modelo principal de cada braço pela regra pré-registrada: maior MCC médio na validação
do braço, desempate pelo ROC-AUC.

Entradas (todas versionadas):
  braço 0: reports/v2/results_{modelo}_n4000.json + reports/v2/teste_cego_antes_{modelo}.json
  braço A: reports/v2/otimizacao/misturado/results_{modelo}_n4000.json + teste_cego_otimizado_misturado_*
  braço B: reports/v2/otimizacao/por_gene/results_{modelo}_n4000.json + teste_cego_otimizado_por_gene_*

Saída: reports/v2/tabelas_comparacao_bracos.tsv (para colar no Google Docs).

Uso (a partir de biomarker_pipeline/):
    python src/relatorios/comparacao_bracos.py
"""

import os
import sys
import json
import logging
from datetime import date

import numpy as np
from scipy.stats import rankdata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

RANDOM_STATE = 42
N_BOOTSTRAP  = 5000
DIM          = 4000
REPORTS_DIR  = "reports/v2"
OUT          = os.path.join(REPORTS_DIR, "tabelas_comparacao_bracos.tsv")

ARMS = [
    ("Braço 0 — linha de base (CV de genes misturados)", "antes", REPORTS_DIR,
     ["svm", "logistic_regression", "random_forest", "gradient_boosting", "naive_bayes", "knn"]),
    ("Braço A — otimizado, CV de genes misturados", "otimizado_misturado", f"{REPORTS_DIR}/otimizacao/misturado",
     ["comite", "logistic_regression", "svm", "trees", "boosting", "mlp"]),
    ("Braço B — otimizado, CV separada por gene", "otimizado_por_gene", f"{REPORTS_DIR}/otimizacao/por_gene",
     ["mlp", "comite", "logistic_regression", "svm", "trees", "boosting"]),
]
LABELS = {
    "logistic_regression": "Regressão Logística", "svm": "SVM", "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting", "naive_bayes": "Naive Bayes", "knn": "kNN",
    "trees": "Árvores (RF/ET)", "boosting": "Boosting (GB/HistGB)", "mlp": "MLP", "comite": "Comitê",
}


def auc_by_ranks(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """ROC-AUC de cada linha (Mann-Whitney com postos médios nos empates); y e p têm forma (B, n)."""
    ranks = rankdata(p, axis=1)
    n_pos = y.sum(axis=1)
    n_neg = y.shape[1] - n_pos
    return ((ranks * y).sum(axis=1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def bootstrap_ci(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """IC 95% do ROC-AUC por gene, reamostrando genes dentro de cada classe."""
    rng = np.random.RandomState(RANDOM_STATE)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    idx = np.hstack([rng.choice(pos, (N_BOOTSTRAP, len(pos))), rng.choice(neg, (N_BOOTSTRAP, len(neg)))])
    lo, hi = np.percentile(auc_by_ranks(y[idx], p[idx]), [2.5, 97.5])
    return float(lo), float(hi)


def pct(x: float) -> str:
    return f"{x * 100:.1f}".replace(".", ",")


def dec(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def main():
    rows = []
    for arm_label, tag, cv_dir, models in ARMS:
        results = {}
        for m in models:
            cv_path = os.path.join(cv_dir, f"results_{m}_n{DIM}.json")
            blind_path = os.path.join(REPORTS_DIR, f"teste_cego_{tag}_{m}.json")
            for path in (cv_path, blind_path):
                if not os.path.exists(path):
                    log.error("%s não encontrado — rode o braço e o teste cego antes (docs/plano_otimizacao.md).", path)
                    sys.exit(1)
            with open(cv_path) as f:
                cv = json.load(f)["metrics"]
            with open(blind_path) as f:
                blind = json.load(f)
            genes = blind["genes"]
            y = np.array([g["label"] for g in genes])
            p = np.array([g["proba_media"] for g in genes])
            lo, hi = bootstrap_ci(y, p)
            results[m] = {"cv": cv, "blind": blind["metricas"]["por_gene"], "limiar": blind["limiar"], "ci": (lo, hi)}
        main_model = max(results, key=lambda m: (results[m]["cv"]["mcc"]["mean"], results[m]["cv"]["roc_auc"]["mean"]))
        for m, r in results.items():
            rows.append((arm_label, m, m == main_model, r))
            log.info("%-45s %-20s CV MCC %.3f | cego ROC-AUC %.3f [%.2f; %.2f] MCC %.3f%s", arm_label, m,
                     r["cv"]["mcc"]["mean"], r["blind"]["roc_auc"], *r["ci"], r["blind"]["mcc"],
                     "  (principal)" if m == main_model else "")

    header = ["Braço", "Modelo", "CV: MCC (%)", "CV: ROC-AUC (%)", "Teste cego: ROC-AUC [IC 95%]",
              "Teste cego: MCC", "Teste cego: sensibilidade", "Teste cego: especificidade", "Limiar"]
    lines = ["Tabela — Comparação dos três braços: validação cruzada de cada braço × teste cego por gene "
             f"(k=3, N={DIM}; 102 genes inéditos, 46 positivos de loci Newly reported de Dand et al., 2025, e 56 "
             f"negativos; IC 95% por bootstrap de genes estratificado por classe, {N_BOOTSTRAP} reamostragens)", "",
             "\t".join(header)]
    prev = None
    for arm_label, m, is_main, r in rows:
        cv, bl = r["cv"], r["blind"]
        lines.append("\t".join([
            arm_label if arm_label != prev else "",
            LABELS[m] + (" (principal)" if is_main else ""),
            f"{pct(cv['mcc']['mean'])} ± {pct(cv['mcc']['std'])}",
            f"{pct(cv['roc_auc']['mean'])} ± {pct(cv['roc_auc']['std'])}",
            f"{dec(bl['roc_auc'])} [{dec(r['ci'][0])}; {dec(r['ci'][1])}]",
            dec(bl["mcc"]), pct(bl["sensitivity"]) + "%", pct(bl["specificity"]) + "%", dec(r["limiar"]),
        ]))
        prev = arm_label
    lines += ["Modelo principal de cada braço: maior MCC médio na validação cruzada do próprio braço, desempate pelo "
              "ROC-AUC (regra pré-registrada). Braços A e B: métricas do teste cego no limiar ajustado na validação.",
              f"Fonte: Elaborado pelos autores ({date.today().year}).", ""]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info("Tabela: %s", OUT)


if __name__ == "__main__":
    main()
