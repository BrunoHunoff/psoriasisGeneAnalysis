# psoriasisGeneAnalysis

Pipeline de identificação de biomarcadores proteicos da psoríase.
Converte sequências FASTA em vetores numéricos via k-mers + Random Projection
e classifica proteínas como candidatas a biomarcador ou não.

---

## Requisitos

- Python 3.10+
- Dependências listadas em `biomarker_pipeline/requirements.txt`

**Linux / macOS (bash):**

```bash
cd biomarker_pipeline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
cd biomarker_pipeline
python -m venv venv
.\venv\Scripts\Activate.ps1     # se der erro de execução: Set-ExecutionPolicy -Scope Process -Bypass
python -m pip install -r requirements.txt
```

> Funciona com Python 3.10 a 3.13. Todas as versões em `requirements.txt` têm wheels prontos para essa faixa, não é preciso compilador.

---

## Estrutura do projeto

```
biomarker_pipeline/
  data/
    raw/                  ← FASTAs dos biomarcadores da psoríase (positivos, label=1)
    raw_negativos/        ← FASTAs do grupo controle (negativos, label=0)
    input/                ← Coloque aqui os FASTAs para predição
    processed/            ← Gerado automaticamente pelo pipeline
  models/                 ← Modelos treinados (gerados localmente)
  reports/                ← Resultados dos experimentos (MDs e JSONs)
  src/
    01_kmer_matrix.py     ← Gera matrizes one-hot k-mer por sequência
    02_project_random.py  ← Random Projection → vetores X.npy + y.npy
    05_train_evaluate.py  ← Avalia modelos via cross-validation
    06_train_final.py     ← Treina modelo final em todos os dados
    07_predict.py         ← Prediz novas proteínas a partir de FASTA
  run_preparation.sh      ← Roda steps 01 e 02 em sequência
  requirements.txt
```

---

## Como rodar do zero

### 1. Preparar os dados e vetores

**Linux / macOS (bash):**

```bash
cd biomarker_pipeline
bash run_preparation.sh
```

**Windows (PowerShell):**

```powershell
cd biomarker_pipeline
.\run_preparation.ps1
```

Gera as matrizes k-mer e os vetores de 1000 componentes para todas as sequências.

### 2. Avaliar os modelos (cross-validation)

```bash
python src/05_train_evaluate.py --model logistic_regression
python src/05_train_evaluate.py --model svm
python src/05_train_evaluate.py --model random_forest
python src/05_train_evaluate.py --model gradient_boosting
python src/05_train_evaluate.py --model naive_bayes
python src/05_train_evaluate.py --model knn
```

Resultados salvos em `reports/results_{modelo}.json`.

### 3. Treinar o modelo final

```bash
python src/06_train_final.py --model logistic_regression
```

Salva o classificador treinado em `models/` para uso na predição.

---

## Predizer novas proteínas

1. Coloque um ou mais arquivos `.fasta` em `data/input/`
2. Execute:

```bash
python src/07_predict.py --model logistic_regression
```

Para predizer um arquivo específico:

```bash
python src/07_predict.py --model logistic_regression --fasta data/input/minha_proteina.fasta
```

O resultado mostra para cada sequência:
- `POSITIVO` ou `NEGATIVO`
- Confiança da predição (%)
- Probabilidade de ser biomarcador da psoríase

---

## Modelos disponíveis

| Modelo | ROC-AUC (exp. 2, N=1000) |
|---|---|
| `logistic_regression` | 99.8% — melhor resultado geral |
| `svm` | 98.7% |
| `random_forest` | 97.9% |
| `gradient_boosting` | 95.0% |
| `knn` | 89.0% |
| `naive_bayes` | 78.1% |

**Recomendado:** `logistic_regression`

---

## Dados de treinamento

- **111 sequências positivas** — 28 proteínas biomarcadoras da psoríase
- **112 sequências negativas** — proteínas housekeeping, não ligadas à psoríase, Artrite Reumatoide e Doença de Crohn
- **Representação:** k-mers de tamanho 3 (vocabulário de 8.000) + Gaussian Random Projection com N=1000
- **Avaliação:** Stratified 10-fold Cross-Validation