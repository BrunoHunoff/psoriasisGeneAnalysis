#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "=== Etapa 1: Perfil vetorial dos biomarcadores ==="

echo "[1/4] Gerando matrizes posicionais k-mer..."
python3 src/01_kmer_matrix.py

echo "[2/4] Projeção SVD global + mean pooling..."
python3 src/02_project_svd.py

echo "[3/4] Projeção Random + mean pooling..."
python3 src/03_project_random.py

echo "[4/4] Comparação SVD vs Random Projection..."
python3 src/04_compare_projections.py

echo ""
echo "=== Concluído ==="
echo "Vetores SVD em:    data/processed/svd/vectors/"
echo "Vetores random em: data/processed/random/vectors/"
echo "Modelos em:        models/"
echo "Gráficos em:       reports/"
