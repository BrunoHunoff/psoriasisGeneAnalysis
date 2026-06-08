# Experimento — Vetor 200 componentes

## Configuração da pipeline de preparação

| Parâmetro | Valor |
|---|---|
| Sequências positivas | 111 (28 proteínas biomarcadoras da psoríase) |
| Sequências negativas | 115 (29 proteínas housekeeping / não ligadas à psoríase) |
| Total de amostras | 226 |
| Representação | Matriz posicional one-hot por sequência |
| Tamanho do k-mer (k) | 3 → vocabulário de 8.000 colunas (20³) |
| Projeção | Gaussian Random Projection |
| Tamanho do vetor | 200 componentes |
| Pooling | Mean pooling sobre as janelas da sequência |
| Avaliação | Stratified 10-fold Cross-Validation |

## Resultados

Ordenado por ROC-AUC decrescente.

| Modelo | Acurácia | F1 | Precisão | Sensibilidade | Especificidade | ROC-AUC | MCC |
|---|---|---|---|---|---|---|---|
| SVM (RBF) | 92.1% ±4.2% | 91.0% | 98.5% | 85.5% | 98.2% | 97.6% | 0.853 |
| Random Forest | 88.5% ±5.2% | 87.4% | 92.5% | 83.7% | 93.0% | 96.2% | 0.778 |
| Logistic Regression | 88.9% ±5.7% | 88.8% | 88.4% | 90.1% | 87.8% | 94.8% | 0.788 |
| Gradient Boosting | 86.4% ±7.1% | 86.0% | 85.5% | 87.3% | 85.3% | 94.4% | 0.733 |
| kNN (k=5) | 73.1% ±9.6% | 76.0% | 69.0% | 85.7% | 61.1% | 81.8% | 0.487 |
| Naive Bayes | 70.8% ±10.5% | 63.2% | 81.9% | 53.3% | 88.0% | 79.6% | 0.447 |
