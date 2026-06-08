# Experimento 2 — Controle dificultado — Vetor 400 componentes

## Configuração da pipeline de preparação

| Parâmetro | Valor |
|---|---|
| Sequências positivas | 111 (28 proteínas biomarcadoras da psoríase) |
| Sequências negativas | 112 (housekeeping + AR + Crohn + não ligadas à psoríase) |
| Total de amostras | 223 |
| Representação | Matriz posicional one-hot por sequência |
| Tamanho do k-mer (k) | 3 → vocabulário de 8.000 colunas (20³) |
| Projeção | Gaussian Random Projection |
| Tamanho do vetor | 400 componentes |
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
| Random Forest | 87.9% ±4.8% | 88.0% | 87.7% | 89.3% | 86.5% | 97.5% | 0.767 |
| SVM (RBF) | 89.2% ±2.9% | 88.8% | 91.0% | 87.3% | 91.1% | 97.1% | 0.790 |
| Logistic Regression | 92.8% ±5.0% | 92.9% | 91.5% | 94.6% | 91.0% | 97.0% | 0.860 |
| Gradient Boosting | 83.8% ±6.5% | 84.1% | 83.1% | 85.5% | 82.2% | 94.7% | 0.681 |
| kNN (k=5) | 70.4% ±8.6% | 76.9% | 63.8% | 97.3% | 43.7% | 89.3% | 0.484 |
| Naive Bayes | 70.4% ±12.3% | 65.0% | 77.9% | 56.9% | 83.7% | 73.5% | 0.426 |
