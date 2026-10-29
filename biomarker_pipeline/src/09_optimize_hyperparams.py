"""
Módulo 09 — Otimização de hiperparâmetros dos braços A e B (docs/plano_otimizacao.md).

Busca bayesiana (Optuna, TPE) sobre os vetores X_n{dim}.npy do 02, com a validação cruzada do braço:
  misturado — RepeatedStratifiedKFold 5×10 (genes misturados, como a linha de base)       → braço A
  por_gene  — 5 repetições de StratifiedGroupKFold 10 folds, grupo = gene (todos os homólogos
              de um gene sempre no mesmo fold)                                               → braço B
Objetivo: MCC médio nas 50 avaliações (5 repetições × 10 folds), com limiar 0,5.

Para cada modelo: busca com o orçamento do plano (+ regra de borda) → avaliação fora do fold
(out-of-fold) do melhor conjunto → limiar de decisão que maximiza o MCC → treino final com todos
os dados. --model comite combina os cinco modelos otimizados do braço (média das probabilidades)
e só pode rodar depois deles.

A busca é retomável: o estudo do Optuna fica num SQLite; rodar o mesmo comando de novo continua
de onde parou. --piloto N roda N tentativas por modelo num diretório separado (otimizacao_piloto/),
sem regra de borda e sem treino final, só para medir o tempo antes da rodada completa.

Saídas (k=3 contínuo; outras representações gravam no subdiretório delas, ver common.dataset_paths):
  reports/v2/otimizacao/{braço}/results_{modelo}_n{dim}.json — melhores hiperparâmetros, métricas, limiar
  reports/v2/otimizacao/{braço}/tentativas_{modelo}.csv       — histórico de todas as tentativas
  reports/v2/otimizacao/{braço}/oof_{modelo}_n{dim}.npz       — probabilidades fora do fold (5 × n)
  reports/v2/otimizacao/{braço}/optuna_{modelo}.db            — estudo do Optuna (gitignored)
  reports/v2/otimizacao/{braço}/tabela_otimizacao.tsv         — resumo do braço, com o modelo principal
  models/v2/otimizacao/{braço}/classifier_{modelo}_n{dim}.pkl + _meta.json

Uso:
    python src/09_optimize_hyperparams.py --arm misturado --model all --n-components 4000
    python src/09_optimize_hyperparams.py --arm por_gene --model svm --n-components 4000
    python src/09_optimize_hyperparams.py --arm por_gene --model comite --n-components 4000
    python src/09_optimize_hyperparams.py --arm misturado --model all --n-components 4000 --piloto 5
"""

import os
import sys
import csv
import json
import math
import time
import pickle
import logging
import argparse
import platform
import subprocess
from datetime import date, datetime

import numpy as np
import optuna
import sklearn
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import (add_dataset_arg, add_representation_args, dataset_paths, patterns_from_args,
                    rep_name, vector_dim, x_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
N_SPLITS     = 10
N_REPEATS    = 5
ARMS = {"misturado": "otimizado_misturado", "por_gene": "otimizado_por_gene"}   # braço → tag do 08
MEMBERS = ["logistic_regression", "svm", "trees", "boosting", "mlp"]
BUDGET = {"logistic_regression": 50, "svm": 150, "trees": 150, "boosting": 100, "mlp": 100}
EDGE_FRACTION     = 0.05   # regra de borda: melhor valor nos 5% extremos da faixa
EDGE_EXTRA_TRIALS = 50
MAX_EXTENSIONS    = 2
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)
METRICS = ("mcc", "roc_auc", "accuracy", "precision", "sensitivity", "specificity", "f1")


# ---------------------------------------------------------------- espaço de busca
def search_space(model: str, p: int) -> list[dict]:
    """
    Espaços da seção 5 do plano. kind: cat | float | int; "hard" = limites válidos usados pela
    regra de borda (None = sem limite); "when" = só existe quando o parâmetro categórico tem o valor.
    """
    if model == "logistic_regression":
        return [
            {"name": "regularizacao", "kind": "cat", "choices": ["l2", "l1", "elasticnet"]},
            {"name": "C", "kind": "float", "low": 1e-5, "high": 1e3, "log": True, "hard": (None, None)},
            {"name": "l1_ratio", "kind": "float", "low": 0.05, "high": 0.95, "log": False, "hard": (0.0, 1.0),
             "when": ("regularizacao", "elasticnet")},
        ]
    if model == "svm":
        return [
            {"name": "kernel", "kind": "cat", "choices": ["rbf", "linear"]},
            {"name": "C", "kind": "float", "low": 2 ** -5, "high": 2 ** 15, "log": True, "hard": (None, None)},
            {"name": "gamma", "kind": "float", "low": 1e-3 / p, "high": 1e2 / p, "log": True, "hard": (None, None),
             "when": ("kernel", "rbf")},
        ]
    if model == "trees":
        return [
            {"name": "tipo", "kind": "cat", "choices": ["random_forest", "extra_trees"]},
            {"name": "max_features", "kind": "int", "low": 1, "high": min(2000, p), "log": True, "hard": (1, p)},
            {"name": "min_samples_leaf", "kind": "int", "low": 1, "high": 10, "log": False, "hard": (1, None)},
            {"name": "max_samples", "kind": "float", "low": 0.2, "high": 1.0, "log": False, "hard": (0.01, 1.0)},
        ]
    if model == "boosting":
        gb, hgb = ("tipo", "gradient_boosting"), ("tipo", "hist_gradient_boosting")
        return [
            {"name": "tipo", "kind": "cat", "choices": ["gradient_boosting", "hist_gradient_boosting"]},
            {"name": "gb_n_estimators", "kind": "int", "low": 100, "high": 2000, "log": False, "hard": (1, None), "when": gb},
            {"name": "gb_learning_rate", "kind": "float", "low": 0.005, "high": 0.5, "log": True, "hard": (None, 1.0), "when": gb},
            {"name": "gb_max_depth", "kind": "int", "low": 1, "high": 6, "log": False, "hard": (1, None), "when": gb},
            {"name": "gb_subsample", "kind": "float", "low": 0.5, "high": 1.0, "log": False, "hard": (0.01, 1.0), "when": gb},
            {"name": "hgb_max_iter", "kind": "int", "low": 100, "high": 2000, "log": False, "hard": (1, None), "when": hgb},
            {"name": "hgb_learning_rate", "kind": "float", "low": 0.005, "high": 0.5, "log": True, "hard": (None, 1.0), "when": hgb},
            {"name": "hgb_max_leaf_nodes", "kind": "int", "low": 4, "high": 64, "log": True, "hard": (2, None), "when": hgb},
            {"name": "hgb_min_samples_leaf", "kind": "int", "low": 5, "high": 50, "log": False, "hard": (1, None), "when": hgb},
            {"name": "hgb_l2_regularization", "kind": "float", "low": 1e-6, "high": 10.0, "log": True, "hard": (None, None), "when": hgb},
            {"name": "hgb_max_features", "kind": "float", "low": 0.05, "high": 1.0, "log": False, "hard": (0.001, 1.0), "when": hgb},
        ]
    if model == "mlp":
        return [
            {"name": "camadas", "kind": "cat", "choices": [1, 2]},
            {"name": "neuronios_1", "kind": "int", "low": 32, "high": 1024, "log": True, "hard": (1, None)},
            {"name": "neuronios_2", "kind": "int", "low": 32, "high": 1024, "log": True, "hard": (1, None),
             "when": ("camadas", 2)},
            {"name": "alpha", "kind": "float", "low": 1e-6, "high": 1e-1, "log": True, "hard": (None, None)},
            {"name": "learning_rate_init", "kind": "float", "low": 1e-4, "high": 1e-2, "log": True, "hard": (None, 1.0)},
            {"name": "lote", "kind": "cat", "choices": [32, 64, 128]},
        ]
    raise ValueError(model)


def initial_bounds(space: list[dict]) -> dict:
    return {s["name"]: [s["low"], s["high"]] for s in space if s["kind"] != "cat"}


def suggest(trial, space: list[dict], bounds: dict) -> dict:
    params = {}
    for s in space:
        cond = s.get("when")
        if cond and params.get(cond[0]) != cond[1]:
            continue
        if s["kind"] == "cat":
            params[s["name"]] = trial.suggest_categorical(s["name"], s["choices"])
        elif s["kind"] == "float":
            lo, hi = bounds[s["name"]]
            params[s["name"]] = trial.suggest_float(s["name"], lo, hi, log=s["log"])
        else:
            lo, hi = bounds[s["name"]]
            params[s["name"]] = trial.suggest_int(s["name"], int(lo), int(hi), log=s["log"])
    return params


def build_estimator(model: str, params: dict):
    """Hiperparâmetros da busca → estimador do scikit-learn (LR, SVM e MLP com StandardScaler)."""
    if model == "logistic_regression":
        # penalty está obsoleto no scikit-learn 1.8: L2 = l1_ratio 0, L1 = l1_ratio 1 (registro de desvios do plano)
        l1_ratio = {"l2": 0.0, "l1": 1.0}.get(params["regularizacao"], params.get("l1_ratio"))
        clf = LogisticRegression(C=params["C"], l1_ratio=l1_ratio, solver="saga", max_iter=5000,
                                 random_state=RANDOM_STATE)
        return Pipeline([("scaler", StandardScaler()), ("model", clf)])
    if model == "svm":
        clf = SVC(kernel=params["kernel"], C=params["C"], gamma=params.get("gamma", "scale"),
                  probability=True, cache_size=500, random_state=RANDOM_STATE)
        return Pipeline([("scaler", StandardScaler()), ("model", clf)])
    if model == "trees":
        cls = RandomForestClassifier if params["tipo"] == "random_forest" else ExtraTreesClassifier
        return cls(n_estimators=1000, max_features=params["max_features"], min_samples_leaf=params["min_samples_leaf"],
                   max_samples=params["max_samples"], bootstrap=True, n_jobs=1, random_state=RANDOM_STATE)
    if model == "boosting":
        if params["tipo"] == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=params["gb_n_estimators"], learning_rate=params["gb_learning_rate"],
                max_depth=params["gb_max_depth"], subsample=params["gb_subsample"], random_state=RANDOM_STATE)
        return HistGradientBoostingClassifier(
            max_iter=params["hgb_max_iter"], learning_rate=params["hgb_learning_rate"],
            max_leaf_nodes=params["hgb_max_leaf_nodes"], min_samples_leaf=params["hgb_min_samples_leaf"],
            l2_regularization=params["hgb_l2_regularization"], max_features=params["hgb_max_features"],
            early_stopping=False, random_state=RANDOM_STATE)
    if model == "mlp":
        layers = (params["neuronios_1"],) if params["camadas"] == 1 else (params["neuronios_1"], params["neuronios_2"])
        clf = MLPClassifier(hidden_layer_sizes=layers, alpha=params["alpha"],
                            learning_rate_init=params["learning_rate_init"], batch_size=params["lote"],
                            activation="relu", solver="adam", early_stopping=True, validation_fraction=0.1,
                            max_iter=500, random_state=RANDOM_STATE)
        return Pipeline([("scaler", StandardScaler()), ("model", clf)])
    raise ValueError(model)


def edge_extensions(space: list[dict], bounds: dict, best: dict) -> dict:
    """
    Regra de borda (seção 5 do plano): faixas cujo melhor valor caiu nos 5% extremos são ampliadas
    naquela direção — 10× nas logarítmicas, uma largura da faixa nas lineares —, sem passar do
    limite válido. Retorna {parâmetro: [novo_min, novo_max]} só para as que mudam.
    """
    out = {}
    for s in space:
        name = s["name"]
        if s["kind"] == "cat" or name not in best:
            continue
        lo, hi = bounds[name]
        v = best[name]
        if s["log"]:
            pos = (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))
        else:
            pos = (v - lo) / (hi - lo)
        hard_lo, hard_hi = s["hard"]
        new_lo, new_hi = lo, hi
        if pos <= EDGE_FRACTION:
            new_lo = lo / 10 if s["log"] else lo - (hi - lo)
            if hard_lo is not None:
                new_lo = max(new_lo, hard_lo)
        if pos >= 1 - EDGE_FRACTION:
            new_hi = hi * 10 if s["log"] else hi + (hi - lo)
            if hard_hi is not None:
                new_hi = min(new_hi, hard_hi)
        if s["kind"] == "int":
            new_lo, new_hi = int(math.floor(new_lo)), int(math.ceil(new_hi))
        if (new_lo, new_hi) != (lo, hi):
            out[name] = [new_lo, new_hi]
    return out


# ---------------------------------------------------------------- validação cruzada
def gene_groups(metadata: dict, n: int) -> np.ndarray:
    """Gene de cada sequência: o nome do FASTA (positivos) ou a parte antes do primeiro "_" (negativos)."""
    return np.array([metadata[str(i)]["protein"].split("_")[0] for i in range(n)])


def make_splits(arm: str, y: np.ndarray, groups: np.ndarray) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """50 divisões (repetição, treino, teste); em cada repetição, cada sequência é teste exatamente uma vez."""
    dummy = np.zeros((len(y), 1))
    if arm == "misturado":
        cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
        return [(i // N_SPLITS, tr, te) for i, (tr, te) in enumerate(cv.split(dummy, y))]
    splits = []
    for r in range(N_REPEATS):
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + r)
        splits += [(r, tr, te) for tr, te in cv.split(dummy, y, groups)]
    return splits


def _fit_predict(estimator, X, y, tr, te):
    est = clone(estimator)
    est.fit(X[tr], y[tr])
    return est.predict_proba(X[te])[:, 1]


def cross_val_probas(estimator, X, y, splits, n_jobs: int) -> list[np.ndarray]:
    return Parallel(n_jobs=n_jobs)(delayed(_fit_predict)(estimator, X, y, tr, te) for _, tr, te in splits)


def metrics_at(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn = int(((y_true == 0) & (pred == 0)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    return {
        "mcc":         float(matthews_corrcoef(y_true, pred)),
        "roc_auc":     float(roc_auc_score(y_true, proba)) if len(set(y_true)) == 2 else float("nan"),
        "accuracy":    float(accuracy_score(y_true, pred)),
        "precision":   float(precision_score(y_true, pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1":          float(f1_score(y_true, pred, zero_division=0)),
    }


def aggregate(per_split: list[dict]) -> dict:
    return {k: {"mean": float(np.nanmean([m[k] for m in per_split])),
                "std": float(np.nanstd([m[k] for m in per_split]))} for k in METRICS}


def split_metrics(y, splits, probas, threshold=0.5) -> dict:
    return aggregate([metrics_at(y[te], p, threshold) for (_, _, te), p in zip(splits, probas)])


def oof_matrix(n: int, splits, probas) -> np.ndarray:
    oof = np.full((N_REPEATS, n), np.nan)
    for (r, _, te), p in zip(splits, probas):
        oof[r, te] = p
    return oof


def choose_threshold(y: np.ndarray, oof: np.ndarray) -> tuple[float, dict]:
    """Limiar que maximiza o MCC médio entre as repetições (empate → o mais próximo de 0,5)."""
    scores = [np.mean([matthews_corrcoef(y, (o >= t).astype(int)) for o in oof]) for t in THRESHOLDS]
    best = max(range(len(THRESHOLDS)), key=lambda i: (round(scores[i], 12), -abs(THRESHOLDS[i] - 0.5)))
    t = float(THRESHOLDS[best])
    return t, aggregate([metrics_at(y, o, t) for o in oof])


# ---------------------------------------------------------------- registro
def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "desconhecido"


def environment() -> dict:
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": np.__version__,
            "optuna": optuna.__version__, "commit": git_commit(), "maquina": platform.node(),
            "cpus": os.cpu_count()}


def fmt(m: dict) -> str:
    return f"{m['mean']*100:.1f} ± {m['std']*100:.1f}"


def write_trials_csv(study, path: str) -> None:
    rows = []
    for t in study.trials:
        row = {"tentativa": t.number, "estado": t.state.name, "mcc_medio": t.value,
               "segundos": t.user_attrs.get("segundos")}
        row.update({f"param_{k}": v for k, v in t.params.items()})
        row.update({k: v for k, v in t.user_attrs.items() if k != "segundos"})
        rows.append(row)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (not k in ("tentativa", "estado", "mcc_medio", "segundos"), k))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- busca de um modelo
def run_search(model, arm, X, y, splits, out_dir, n_jobs, budget, use_edge_rule, p):
    space = search_space(model, p)
    storage = f"sqlite:///{os.path.join(out_dir, f'optuna_{model}.db')}"
    study = optuna.create_study(study_name=f"{arm}_{model}", storage=storage, direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE), load_if_exists=True)
    if "faixas" not in study.user_attrs:
        study.set_user_attr("faixas", initial_bounds(space))
        study.set_user_attr("extensoes", [])

    def objective(trial):
        params = suggest(trial, space, study.user_attrs["faixas"])
        t0 = time.time()
        probas = cross_val_probas(build_estimator(model, params), X, y, splits, n_jobs)
        m = split_metrics(y, splits, probas)
        trial.set_user_attr("segundos", round(time.time() - t0, 1))
        for k in METRICS:
            trial.set_user_attr(f"{k}_mean", m[k]["mean"])
            trial.set_user_attr(f"{k}_std", m[k]["std"])
        return m["mcc"]["mean"]

    def report(study, trial):
        best = study.best_value if any(t.value is not None for t in study.trials) else float("nan")
        log.info("[%s/%s] tentativa %3d  %-8s MCC %.4f  (melhor %.4f)  %5.0f s  %s", arm, model, trial.number,
                 trial.state.name, trial.value if trial.value is not None else float("nan"), best,
                 trial.user_attrs.get("segundos", 0), trial.params)

    while True:
        extensions = study.user_attrs["extensoes"]
        target = budget + EDGE_EXTRA_TRIALS * len(extensions)
        done = len([t for t in study.trials if t.state.is_finished()])
        if done < target:
            log.info("[%s/%s] %d de %d tentativas feitas; rodando %d", arm, model, done, target, target - done)
            study.optimize(objective, n_trials=target - done, callbacks=[report], catch=(Exception,))
        if not use_edge_rule or len(extensions) >= MAX_EXTENSIONS:
            break
        ext = edge_extensions(space, study.user_attrs["faixas"], study.best_trial.params)
        if not ext:
            break
        bounds = dict(study.user_attrs["faixas"])
        bounds.update(ext)
        study.set_user_attr("faixas", bounds)
        study.set_user_attr("extensoes", extensions + [{"apos_tentativas": target, "faixas": ext}])
        log.info("[%s/%s] regra de borda: ampliando %s e rodando mais %d tentativas", arm, model, ext, EDGE_EXTRA_TRIALS)
    return study


def finalize_model(model, arm, study, X, y, splits, out_dir, models_dir, n_jobs, rep, patterns, dim, budget, t_start):
    best = study.best_trial
    estimator = build_estimator(model, best.params)
    log.info("[%s/%s] melhor tentativa %d (MCC %.4f): %s — avaliando fora do fold e treinando o modelo final",
             arm, model, best.number, best.value, best.params)
    probas = cross_val_probas(estimator, X, y, splits, n_jobs)
    oof = oof_matrix(len(y), splits, probas)
    np.savez_compressed(os.path.join(out_dir, f"oof_{model}_n{dim}.npz"), oof=oof, y=y)
    metrics = split_metrics(y, splits, probas)
    threshold, metrics_t = choose_threshold(y, oof)

    final = clone(estimator).fit(X, y)
    clf_path = os.path.join(models_dir, f"classifier_{model}_n{dim}.pkl")
    with open(clf_path, "wb") as f:
        pickle.dump(final, f)

    finished = [t for t in study.trials if t.state.is_finished()]
    report = {
        "model": model, "braco": arm, "tag": ARMS[arm], "dataset": "tcc2",
        "representation": rep, "patterns": list(patterns), "projection": True, "n_components": dim,
        "validacao": ("RepeatedStratifiedKFold 5×10" if arm == "misturado"
                      else "StratifiedGroupKFold 10 folds × 5 repetições, grupo = gene"),
        "objetivo": "MCC médio nas 50 avaliações, limiar 0,5",
        "orcamento": budget, "tentativas": len(finished),
        "tentativas_com_falha": len([t for t in finished if t.state.name == "FAIL"]),
        "extensoes_de_faixa": study.user_attrs["extensoes"], "faixas_finais": study.user_attrs["faixas"],
        "melhor_tentativa": best.number, "best_params": best.params,
        "estimador": repr(estimator),
        "metrics": metrics, "limiar": threshold, "metrics_fora_do_fold_no_limiar": metrics_t,
        "ambiente": environment(), "concluido_em": datetime.now().isoformat(timespec="seconds"),
        "duracao_horas": round((time.time() - t_start) / 3600, 2),
    }
    with open(os.path.join(out_dir, f"results_{model}_n{dim}.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    meta = {"model": model, "braco": arm, "tag": ARMS[arm], "dataset": "tcc2", "representation": rep,
            "patterns": list(patterns), "projection": True, "random_state": RANDOM_STATE, "n_components": dim,
            "best_params": best.params, "limiar": threshold, "n_train": int(len(y)),
            "classifier_path": clf_path}
    with open(clf_path.replace(".pkl", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    write_trials_csv(study, os.path.join(out_dir, f"tentativas_{model}.csv"))
    log.info("[%s/%s] CV: MCC %s, ROC-AUC %s | limiar %.2f → MCC fora do fold %s", arm, model,
             fmt(metrics["mcc"]), fmt(metrics["roc_auc"]), threshold, fmt(metrics_t["mcc"]))


def finalize_committee(arm, X, y, splits, out_dir, models_dir, dim, rep, patterns):
    """Comitê: média das probabilidades fora do fold e dos modelos finais dos cinco membros."""
    oofs = []
    for m in MEMBERS:
        path = os.path.join(out_dir, f"oof_{m}_n{dim}.npz")
        if not os.path.exists(path) or not os.path.exists(os.path.join(models_dir, f"classifier_{m}_n{dim}.pkl")):
            log.error("Membro %s ainda não otimizado no braço %s (%s) — rode os cinco modelos antes do comitê.",
                      m, arm, path)
            sys.exit(1)
        oofs.append(np.load(path)["oof"])
    oof = np.mean(oofs, axis=0)
    np.savez_compressed(os.path.join(out_dir, f"oof_comite_n{dim}.npz"), oof=oof, y=y)
    probas = [oof[r, te] for r, _, te in splits]
    metrics = split_metrics(y, splits, probas)
    threshold, metrics_t = choose_threshold(y, oof)
    report = {"model": "comite", "braco": arm, "tag": ARMS[arm], "dataset": "tcc2", "representation": rep,
              "patterns": list(patterns), "projection": True, "n_components": dim, "membros": MEMBERS,
              "combinacao": "média simples das probabilidades", "metrics": metrics, "limiar": threshold,
              "metrics_fora_do_fold_no_limiar": metrics_t, "ambiente": environment(),
              "concluido_em": datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(out_dir, f"results_comite_n{dim}.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    meta = {"model": "comite", "braco": arm, "tag": ARMS[arm], "dataset": "tcc2", "representation": rep,
            "patterns": list(patterns), "projection": True, "random_state": RANDOM_STATE, "n_components": dim,
            "membros": MEMBERS, "limiar": threshold, "n_train": int(len(y))}
    with open(os.path.join(models_dir, f"classifier_comite_n{dim}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log.info("[%s/comite] CV: MCC %s, ROC-AUC %s | limiar %.2f → MCC fora do fold %s", arm,
             fmt(metrics["mcc"]), fmt(metrics["roc_auc"]), threshold, fmt(metrics_t["mcc"]))


def write_summary(out_dir: str, arm: str, dim: int) -> None:
    """Tabela do braço; modelo principal = maior MCC médio na CV do braço, desempate ROC-AUC (seção 8 do plano)."""
    labels = {"logistic_regression": "Regressão Logística", "svm": "SVM", "trees": "Árvores (RF/ET)",
              "boosting": "Boosting (GB/HistGB)", "mlp": "MLP", "comite": "Comitê"}
    results = {}
    for m in MEMBERS + ["comite"]:
        path = os.path.join(out_dir, f"results_{m}_n{dim}.json")
        if os.path.exists(path):
            with open(path) as f:
                results[m] = json.load(f)
    if not results:
        return
    main = max(results, key=lambda m: (results[m]["metrics"]["mcc"]["mean"], results[m]["metrics"]["roc_auc"]["mean"]))
    pct = lambda d: f"{d['mean']*100:.1f}% ± {d['std']*100:.1f}%".replace(".", ",")
    arm_label = {"misturado": "braço A — CV de genes misturados", "por_gene": "braço B — CV separada por gene"}[arm]
    lines = [f"Tabela — Otimização de hiperparâmetros, {arm_label} (k=3, N={dim}; média ± desvio padrão nas "
             f"50 avaliações da validação 5×10 do braço; modelo principal = maior MCC)", "",
             "\t".join(["Modelo", "MCC", "ROC-AUC", "Sensibilidade", "Especificidade", "Limiar ajustado",
                        "MCC fora do fold no limiar", "Tentativas", "Melhores hiperparâmetros"])]
    for m, r in results.items():
        mt = r["metrics"]
        lines.append("\t".join([labels[m] + (" (principal)" if m == main else ""), pct(mt["mcc"]), pct(mt["roc_auc"]),
                                pct(mt["sensitivity"]), pct(mt["specificity"]), f"{r['limiar']:.2f}".replace(".", ","),
                                pct(r["metrics_fora_do_fold_no_limiar"]["mcc"]), str(r.get("tentativas", "—")),
                                json.dumps(r.get("best_params", {"membros": r.get("membros")}), ensure_ascii=False)]))
    lines += [f"Fonte: Elaborado pelos autores ({date.today().year}).", ""]
    with open(os.path.join(out_dir, "tabela_otimizacao.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Otimização de hiperparâmetros dos braços A e B.")
    parser.add_argument("--arm", required=True, choices=list(ARMS), help="misturado (braço A) ou por_gene (braço B).")
    parser.add_argument("--model", required=True, choices=MEMBERS + ["comite", "all"],
                        help="Modelo a otimizar; all = os cinco e depois o comitê.")
    parser.add_argument("--piloto", type=int, default=None, metavar="N",
                        help="Roda só N tentativas por modelo em otimizacao_piloto/, para medir tempo.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Processos paralelos (padrão: todos os núcleos).")
    add_dataset_arg(parser)
    add_representation_args(parser, require_n_components=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset != "tcc2":
        log.error("A otimização é do TCC II (docs/plano_otimizacao.md): use --dataset tcc2.")
        sys.exit(1)
    if not args.n_components:
        log.error("A otimização usa vetores projetados (o plano fixa N=4000); --n-components 0 não é aceito.")
        sys.exit(1)
    patterns = patterns_from_args(args)
    dim = vector_dim(patterns, args.n_components)
    rep = rep_name(patterns)
    paths = dataset_paths(args.dataset, patterns)
    sub = "otimizacao_piloto" if args.piloto else "otimizacao"
    out_dir = os.path.join(paths["reports_dir"], sub, args.arm)
    models_dir = os.path.join(paths["models_dir"], sub, args.arm)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    in_x = x_path(paths["proc_dir"], dim)
    meta_path = os.path.join(paths["proc_dir"], "metadata.json")
    for path in (in_x, meta_path):
        if not os.path.exists(path):
            log.error("%s não encontrado — rode 01_kmer_matrix.py e 02_project_random.py --dataset tcc2 "
                      "--n-components %d antes.", path, args.n_components)
            sys.exit(1)
    X = np.load(in_x)
    y = np.load(os.path.join(paths["proc_dir"], "y.npy")).astype(int)
    with open(meta_path) as f:
        groups = gene_groups(json.load(f), len(y))
    splits = make_splits(args.arm, y, groups)
    log.info("Braço %s (%s): X=%s, %d genes, %d divisões de validação | piloto=%s | ambiente %s",
             args.arm, ARMS[args.arm], X.shape, len(set(groups)), len(splits), args.piloto, environment())

    models = MEMBERS + ["comite"] if args.model == "all" else [args.model]
    for model in models:
        if model == "comite":
            if args.piloto:
                continue
            finalize_committee(args.arm, X, y, splits, out_dir, models_dir, dim, rep, patterns)
            continue
        t_start = time.time()
        budget = args.piloto or BUDGET[model]
        study = run_search(model, args.arm, X, y, splits, out_dir, args.n_jobs, budget,
                           use_edge_rule=not args.piloto, p=dim)
        if args.piloto:
            secs = [t.user_attrs.get("segundos", 0) for t in study.trials if t.state.name == "COMPLETE"]
            if not secs:
                log.error("[piloto %s/%s] nenhuma tentativa terminou — veja os erros acima.", args.arm, model)
                continue
            log.info("[piloto %s/%s] %d tentativas, %.0f s em média por tentativa (máx %.0f s) → orçamento de %d "
                     "tentativas ≈ %.1f h", args.arm, model, len(secs), np.mean(secs), np.max(secs), BUDGET[model],
                     np.mean(secs) * BUDGET[model] / 3600)
            write_trials_csv(study, os.path.join(out_dir, f"tentativas_{model}.csv"))
            continue
        finalize_model(model, args.arm, study, X, y, splits, out_dir, models_dir, args.n_jobs, rep, patterns, dim,
                       BUDGET[model], t_start)
    if not args.piloto:
        write_summary(out_dir, args.arm, dim)
    log.info("Pronto: %s", out_dir)


if __name__ == "__main__":
    main()
