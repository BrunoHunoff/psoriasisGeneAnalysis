# Experimento 2 — Controle dificultado — Vetor 300 componentes

## Configuração da pipeline de preparação

| Parâmetro | Valor |
|---|---|
| Sequências positivas | 111 (28 proteínas biomarcadoras da psoríase) |
| Sequências negativas | 112 (housekeeping + AR + Crohn + não ligadas à psoríase) |
| Total de amostras | 223 |
| Representação | Matriz posicional one-hot por sequência |
| Tamanho do k-mer (k) | 3 → vocabulário de 8.000 colunas (20³) |
| Projeção | Gaussian Random Projection |
| Tamanho do vetor | 300 componentes |
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
| Random Forest | 90.6% ±5.1% | 90.3% | 92.5% | 89.2% | 91.8% | 98.2% | 0.820 |
| Logistic Regression | 92.4% ±5.3% | 92.5% | 91.5% | 93.7% | 91.0% | 97.3% | 0.850 |
| SVM (RBF) | 89.7% ±3.5% | 89.2% | 91.4% | 88.3% | 91.0% | 97.3% | 0.802 |
| Gradient Boosting | 86.5% ±6.7% | 87.0% | 85.0% | 90.0% | 82.9% | 94.9% | 0.739 |
| kNN (k=5) | 70.4% ±6.8% | 76.8% | 63.6% | 97.3% | 43.7% | 86.8% | 0.486 |
| Naive Bayes | 70.4% ±12.3% | 65.2% | 77.1% | 57.8% | 82.8% | 73.2% | 0.425 |
