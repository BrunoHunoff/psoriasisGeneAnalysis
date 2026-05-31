#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "=== Pipeline do modelo ==="

echo "[1/3] Treinando modelos (SVD + Random, todas as abordagens)..."
python3 src/05_train_model.py --projection both

echo "[2/3] Avaliando modelos (Leave-One-Out)..."
python3 src/06_evaluate_model.py

echo "[3/3] Comparando e selecionando melhor modelo..."
python3 src/07_compare_models.py

echo ""
echo "=== Modelo concluído ==="
echo "Modelos treinados: models/"
echo "Resultados:        reports/model_comparison.csv"
echo "Melhor config:     models/best_model_config.json"
