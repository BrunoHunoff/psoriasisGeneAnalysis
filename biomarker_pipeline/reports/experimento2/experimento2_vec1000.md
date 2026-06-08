# Experimento 2 — Controle dificultado — Vetor 1000 componentes

## Configuração da pipeline de preparação

| Parâmetro | Valor |
|---|---|
| Sequências positivas | 111 (28 proteínas biomarcadoras da psoríase) |
| Sequências negativas | 112 (housekeeping + AR + Crohn + não ligadas à psoríase) |
| Total de amostras | 223 |
| Representação | Matriz posicional one-hot por sequência |
| Tamanho do k-mer (k) | 3 → vocabulário de 8.000 colunas (20³) |
| Projeção | Gaussian Random Projection |
| Tamanho do vetor | 1000 componentes |
| Pooling | Mean pooling sobre as janelas da sequência |
| Avaliação | Stratified 10-fold Cross-Validation |

## Grupo controle (negativos)

| Categoria | Proteínas |
|---|---|
| Housekeeping (fácil) | ACTB, ALB, ALDOA, ATP5F1A, CAT, COL1A1, ENO1, GAPDH, HBA1, HBB, INS, LDHA, PGK1, PKM, SOD1, TPI1, TUBB, VIM |
| Não ligadas à psoríase (difícil) | CD3E, IFNG, IL10, IL2, IL5 |
| Artrite Reumatoide (difícil) | CTLA4, PADI4, PTPN22, TRAF1 |
| Doença de Crohn (difícil) | ATG16L1, IRGM, NOD2 |

## Resultados

Ordenado por ROC-AUC decrescente.

| Modelo | Acurácia | F1 | Precisão | Sensibilidade | Especificidade | ROC-AUC | MCC |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 97.8% ±2.2% | 97.8% | 97.6% | 98.2% | 97.3% | 99.8% | 0.957 |
| SVM (RBF) | 91.9% ±3.4% | 91.4% | 96.3% | 87.3% | 96.4% | 98.7% | 0.844 |
| Random Forest | 93.3% ±2.9% | 93.1% | 95.4% | 91.0% | 95.6% | 97.9% | 0.868 |
| Gradient Boosting | 86.6% ±7.1% | 87.1% | 84.7% | 90.2% | 83.0% | 95.0% | 0.740 |
| kNN (k=5) | 68.6% ±6.1% | 76.2% | 61.7% | 100.0% | 37.4% | 89.0% | 0.476 |
| Naive Bayes | 72.2% ±11.4% | 66.2% | 82.1% | 56.9% | 87.3% | 78.1% | 0.470 |
