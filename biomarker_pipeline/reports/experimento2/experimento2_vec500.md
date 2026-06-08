# Experimento 2 — Controle dificultado — Vetor 500 componentes

## Configuração da pipeline de preparação

| Parâmetro | Valor |
|---|---|
| Sequências positivas | 111 (28 proteínas biomarcadoras da psoríase) |
| Sequências negativas | 112 (housekeeping + AR + Crohn + não ligadas à psoríase) |
| Total de amostras | 223 |
| Representação | Matriz posicional one-hot por sequência |
| Tamanho do k-mer (k) | 3 → vocabulário de 8.000 colunas (20³) |
| Projeção | Gaussian Random Projection |
| Tamanho do vetor | 500 componentes |
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
| Logistic Regression | 92.4% ±5.3% | 92.5% | 90.9% | 94.6% | 90.1% | 98.6% | 0.852 |
| Random Forest | 91.9% ±4.4% | 92.0% | 91.6% | 92.8% | 91.1% | 98.3% | 0.843 |
| SVM (RBF) | 91.0% ±4.0% | 90.5% | 94.7% | 87.3% | 94.6% | 98.0% | 0.829 |
| Gradient Boosting | 84.7% ±5.4% | 85.0% | 83.2% | 87.3% | 82.0% | 95.8% | 0.699 |
| kNN (k=5) | 70.8% ±5.0% | 77.5% | 63.3% | 100.0% | 41.9% | 90.3% | 0.513 |
| Naive Bayes | 70.4% ±11.8% | 64.9% | 78.2% | 56.9% | 83.7% | 74.6% | 0.428 |
