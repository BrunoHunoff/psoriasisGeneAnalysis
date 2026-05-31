#!/bin/bash
set -e
echo "=== Pipeline de preparação — SVD + Random Projection ==="

# Ativa o venv se existir
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "[1/4] Gerando matrizes posicionais k-mer..."
python src/01_kmer_matrix.py

echo "[2/4] Projeção SVD..."
python src/02_project_svd.py

echo "[3/4] Projeção Random..."
python src/03_project_random.py

echo "[4/4] Comparação das projeções..."
python src/04_compare_projections.py

echo ""
echo "=== Preparação concluída ==="
echo "Vetores SVD:    data/processed/svd/vectors/"
echo "Vetores Random: data/processed/random/vectors/"
echo "Modelos:        models/"
echo "Gráficos:       reports/"
