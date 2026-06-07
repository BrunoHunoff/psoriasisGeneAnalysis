#!/bin/bash
set -e
echo "=== Pipeline de preparação — Random Projection ==="

# Ativa o venv se existir
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "[1/2] Gerando matrizes posicionais k-mer..."
python src/01_kmer_matrix.py

echo "[2/2] Projeção Random + vetores finais..."
python src/02_project_random.py

echo ""
echo "=== Preparação concluída ==="
echo "Arrays para classificação:"
echo "  data/processed/X.npy        — (n_seq, 500)"
echo "  data/processed/y.npy        — (n_seq,) labels 1/0"
echo "  data/processed/metadata.json"
echo "Vetores individuais: data/processed/vectors/"
echo "Modelo:              models/random_projection.pkl"
