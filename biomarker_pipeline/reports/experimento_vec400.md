# Experimento — Vetor 400 componentes

## Configuração da pipeline de preparação

| Parâmetro | Valor |
|---|---|
| Sequências positivas | 111 (28 proteínas biomarcadoras da psoríase) |
| Sequências negativas | 115 (29 proteínas housekeeping / não ligadas à psoríase) |
| Total de amostras | 226 |
| Representação | Matriz posicional one-hot por sequência |
| Tamanho do k-mer (k) | 3 → vocabulário de 8.000 colunas (20³) |
| Projeção | Gaussian Random Projection |
| Tamanho do vetor | 400 componentes |
| Pooling | Mean pooling sobre as janelas da sequência |
| Avaliação | Stratified 10-fold Cross-Validation |

## Resultados

Ordenado por ROC-AUC decrescente.

| Modelo | Acurácia | F1 | Precisão | Sensibilidade | Especificidade | ROC-AUC | MCC |
|---|---|---|---|---|---|---|---|
| Random Forest | 89.8% ±4.4% | 88.7% | 95.3% | 83.8% | 95.6% | 97.9% | 0.807 |
| SVM (RBF) | 89.5% ±5.1% | 88.7% | 94.1% | 84.6% | 93.9% | 97.6% | 0.798 |
| Gradient Boosting | 88.9% ±5.4% | 88.2% | 91.3% | 86.4% | 91.3% | 97.4% | 0.786 |
| Logistic Regression | 90.8% ±5.7% | 91.0% | 88.1% | 94.6% | 87.0% | 96.2% | 0.823 |
| kNN (k=5) | 78.0% ±7.7% | 76.9% | 79.2% | 76.6% | 79.2% | 85.7% | 0.573 |
| Naive Bayes | 73.9% ±11.8% | 67.4% | 84.0% | 57.8% | 89.5% | 82.2% | 0.503 |
