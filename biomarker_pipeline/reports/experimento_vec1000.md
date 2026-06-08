# Experimento — Vetor 1000 componentes

## Configuração da pipeline de preparação

| Parâmetro | Valor |
|---|---|
| Sequências positivas | 111 (28 proteínas biomarcadoras da psoríase) |
| Sequências negativas | 115 (29 proteínas housekeeping / não ligadas à psoríase) |
| Total de amostras | 226 |
| Representação | Matriz posicional one-hot por sequência |
| Tamanho do k-mer (k) | 3 → vocabulário de 8.000 colunas (20³) |
| Projeção | Gaussian Random Projection |
| Tamanho do vetor | 1000 componentes |
| Pooling | Mean pooling sobre as janelas da sequência |
| Avaliação | Stratified 10-fold Cross-Validation |

## Resultados

Ordenado por ROC-AUC decrescente.

| Modelo | Acurácia | F1 | Precisão | Sensibilidade | Especificidade | ROC-AUC | MCC |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 98.7% ±2.0% | 98.7% | 98.4% | 99.1% | 98.2% | 99.8% | 0.974 |
| SVM (RBF) | 93.0% ±3.5% | 92.2% | 99.2% | 86.4% | 99.1% | 98.7% | 0.867 |
| Random Forest | 93.0% ±3.9% | 92.7% | 94.3% | 91.9% | 93.9% | 98.0% | 0.865 |
| Gradient Boosting | 90.2% ±7.7% | 90.0% | 89.8% | 91.0% | 89.5% | 96.4% | 0.809 |
| kNN (k=5) | 74.3% ±8.9% | 69.4% | 82.0% | 63.3% | 85.2% | 87.6% | 0.516 |
| Naive Bayes | 78.3% ±10.9% | 70.7% | 96.0% | 57.9% | 98.3% | 86.2% | 0.615 |
