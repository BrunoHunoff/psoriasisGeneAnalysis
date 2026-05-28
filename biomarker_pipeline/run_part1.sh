#!/bin/bash
set -e
cd "$(dirname "$0")"

# Ativa o venv se existir
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "=== Etapa 1: Perfil vetorial dos biomarcadores ==="

echo "[1/3] Gerando matrizes posicionais k-mer..."
python3 src/01_kmer_matrix.py

echo "[2/3] SVD global + mean pooling..."
python3 src/02_svd_global.py

echo "[3/3] Construindo perfil do biomarcador..."
python3 src/03_profile.py

echo ""
echo "=== Concluído ==="
echo "Perfil salvo em:  data/processed/profile_mean.npy"
echo "Modelo SVD em:    models/svd_global.pkl"
echo "Vetores em:       data/processed/vectors/"
echo "Gráficos em:      reports/"
