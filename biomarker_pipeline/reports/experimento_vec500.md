# Experimento 01 — Baseline (vetor 500 componentes)

## Configuração da pipeline de preparação

| Parâmetro | Valor |
|---|---|
| Sequências positivas | 111 (28 proteínas biomarcadoras da psoríase) |
| Sequências negativas | 115 (29 proteínas housekeeping / não ligadas à psoríase) |
| Total de amostras | 226 |
| Representação | Matriz posicional one-hot por sequência |
| Tamanho do k-mer (k) | 3 → vocabulário de 8.000 colunas (20³) |
| Projeção | Gaussian Random Projection |
| Tamanho do vetor | 500 componentes |
| Pooling | Mean pooling sobre as janelas da sequência |
| Avaliação | Stratified 10-fold Cross-Validation |

## Resultados

Ordenado por ROC-AUC decrescente.

| Modelo | Acurácia | F1 | Precisão | Sensibilidade | Especificidade | ROC-AUC | MCC |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 91.7% ±5.3% | 91.9% | 88.8% | 95.5% | 87.9% | 99.0% | 0.839 |
| SVM (RBF) | 92.9% ±2.9% | 92.2% | 98.4% | 87.3% | 98.2% | 98.4% | 0.868 |
| Random Forest | 91.2% ±2.7% | 90.7% | 93.9% | 88.3% | 93.8% | 96.7% | 0.829 |
| Gradient Boosting | 86.8% ±5.8% | 86.0% | 87.6% | 85.5% | 87.7% | 95.4% | 0.741 |
| kNN (k=5) | 78.8% ±9.0% | 77.3% | 81.1% | 75.8% | 81.8% | 86.9% | 0.594 |
| Naive Bayes | 75.7% ±12.4% | 69.1% | 86.2% | 59.6% | 91.4% | 83.1% | 0.542 |
