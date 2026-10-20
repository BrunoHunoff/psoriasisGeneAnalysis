# Pipeline de Classificação de Proteínas Associadas à Psoríase

TCC de Engenharia de Software — Universidade Positivo (2026).
Pipeline de classificação binária de proteínas baseada em representação por
k-mers de aminoácidos, redução por Gaussian Random Projection e seis
modelos supervisionados avaliados via validação cruzada estratificada de
10 folds com otimização de hiperparâmetros por `GridSearchCV`.

---

## Visão geral

| Etapa | Script | Saída |
|------|--------|-------|
| 1. Matriz posicional one-hot k-mer (k=3, vocab=8.000) | `src/01_kmer_matrix.py` | `data/processed/matrices/*.npy`, `sequence_registry.json`, `kmer_vocab.json` |
| 2. Gaussian Random Projection + mean pooling | `src/02_project_random.py` | `data/processed/X.npy`, `y.npy`, `metadata.json`, `models/random_projection.pkl` |
| 3. Avaliação 10-fold CV + GridSearchCV | `src/05_train_evaluate.py` | `reports/results_{modelo}.json`, `models/classifier_{modelo}_n{N}.pkl` |
| 4. Treino final em todos os dados | `src/06_train_final.py` | `models/classifier_{modelo}_n{N}.pkl`, `_meta.json` |
| 5. Predição de proteínas novas | `src/07_predict.py` | saída no terminal |

Scripts de orquestração:

- `src/run_experiments.py` — roda 02 + 05 para `N ∈ {300, 400, 500, 1000}` × 6 modelos e gera `reports/tabelas_resultados.tsv`.
- `src/run_predictions.py` — roda 07 sobre as 5 proteínas de validação externa (INS, PTPN22, IL12B, CARD14, RUNX3) e gera `reports/tabelas_predicoes.tsv`.

---

## Dataset

- **Positivos** (`data/raw/`): 111 sequências dos 28 genes notáveis dos 21 loci confirmados por Tsoi et al. (2012) — incluindo HLA-C, IL12B, IL23R, TRAF3IP2, TNFAIP3, TNIP1, entre outros.
- **Negativos** (`data/raw_negativos/`): 112 sequências organizadas em três níveis de dificuldade discriminativa:
  - housekeeping (metabolismo, estrutura celular);
  - imunes sem associação genética com psoríase;
  - marcadores de doença de Crohn e artrite reumatoide (sobreposição parcial de vias biológicas).

Todas as sequências foram obtidas do UniProt/Swiss-Prot (entradas Reviewed), compostas exclusivamente pelos 20 aminoácidos canônicos.

---

## Modelos avaliados

Logistic Regression, SVM (kernel RBF), Random Forest, Gradient Boosting, k-NN e Naive Bayes Gaussiano — todos implementados via `scikit-learn`.

Métricas reportadas: acurácia, precisão, sensibilidade, especificidade, F1-score, ROC-AUC e MCC.

---

## Instalação

```bash
python3 -m venv venv
source venv/bin/activate            # Linux/macOS
# venv\Scripts\activate              # Windows
pip install -r requirements.txt
```

---

## Execução

Todos os scripts recebem `--dataset tcc1` (dataset congelado do TCC I) ou `--dataset tcc2`
(expansão Dand et al., 2025 — padrão). Cada dataset grava em diretórios próprios
(`data/processed` / `models` / `reports` para tcc1; `data/processed_v2` / `models/v2` /
`reports/v2` para tcc2), então um nunca sobrescreve o outro.

### Preparação (matrizes k-mer + projeção)

```bash
bash run_preparation.sh tcc2         # Linux/macOS  (args: dataset, N — padrão tcc2 1000)
# .\run_preparation.ps1 tcc2         # Windows
```

### Experimentos completos (4 valores de N × 6 modelos)

```bash
python src/run_experiments.py --dataset tcc2
```

### Treino final (hiperparâmetros vencedores do GridSearchCV) e predição de uma proteína nova

```bash
python src/06_train_final.py --model svm --dataset tcc2
python src/07_predict.py --model svm --dataset tcc2 --fasta data/input/sua_proteina.fasta
```

### Validação externa em lote (5 proteínas do TCC I)

```bash
python src/run_predictions.py --dataset tcc2
```

### Teste cego do TCC II — rodar UMA vez, com o modelo final já escolhido

```bash
python src/08_blind_test.py --model all
```

---

## Saídas

| Caminho | Conteúdo |
|---------|----------|
| `reports/results_{modelo}_n{N}.json` | Métricas (média ± desvio) de cada combinação modelo × N |
| `reports/tabelas_resultados.tsv` | Tabelas consolidadas em TSV (Google Docs/Word) |
| `reports/tabelas_predicoes.tsv` | Tabela de validação externa |
| `reports/experimento2/experimento2_vec*.md` | Relatórios detalhados por valor de N |
| `models/classifier_{modelo}_n{N}.pkl` | Classificadores treinados |
| `models/random_projection.pkl` | Projeção Gaussiana ajustada |

---

## Estrutura do repositório

```
biomarker_pipeline/
├── data/
│   ├── raw/              # FASTAs positivos (psoríase)
│   ├── raw_negativos/    # FASTAs negativos (controle)
│   ├── input/            # FASTAs para predição
│   └── processed/        # Matrizes, vetores e arrays X/y
├── src/                  # Pipeline (01, 02, 05, 06, 07 + orquestradores)
├── models/               # Modelos serializados
├── reports/              # Resultados e tabelas
├── notebooks/            # EDA, scree plot, resultados
├── requirements.txt
└── run_preparation.{sh,ps1}
```
