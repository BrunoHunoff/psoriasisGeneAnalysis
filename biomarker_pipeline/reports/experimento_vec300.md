# Experimento — Vetor 300 componentes

## Configuração da pipeline de preparação

| Parâmetro | Valor |
|---|---|
| Sequências positivas | 111 (28 proteínas biomarcadoras da psoríase) |
| Sequências negativas | 115 (29 proteínas housekeeping / não ligadas à psoríase) |
| Total de amostras | 226 |
| Representação | Matriz posicional one-hot por sequência |
| Tamanho do k-mer (k) | 3 → vocabulário de 8.000 colunas (20³) |
| Projeção | Gaussian Random Projection |
| Tamanho do vetor | 300 componentes |
| Pooling | Mean pooling sobre as janelas da sequência |
| Avaliação | Stratified 10-fold Cross-Validation |

## Resultados

Ordenado por ROC-AUC decrescente.

| Modelo | Acurácia | F1 | Precisão | Sensibilidade | Especificidade | ROC-AUC | MCC |
|---|---|---|---|---|---|---|---|
| SVM (RBF) | 91.2% ±3.3% | 90.4% | 96.0% | 86.4% | 95.5% | 98.1% | 0.835 |
| Logistic Regression | 91.6% ±5.0% | 91.7% | 89.5% | 94.5% | 88.7% | 96.9% | 0.839 |
| Random Forest | 87.6% ±4.2% | 86.8% | 91.3% | 83.7% | 91.3% | 96.5% | 0.762 |
| Gradient Boosting | 89.0% ±4.0% | 88.4% | 89.8% | 88.2% | 89.5% | 96.4% | 0.787 |
| Naive Bayes | 74.3% ±12.1% | 67.9% | 84.9% | 57.8% | 90.5% | 82.0% | 0.513 |
| kNN (k=5) | 72.7% ±10.3% | 73.0% | 70.9% | 77.4% | 68.0% | 80.6% | 0.470 |
