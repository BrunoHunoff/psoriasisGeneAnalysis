#!/bin/bash
# Uso: bash run_preparation.sh [tcc1|tcc2] [N]     (padrão: tcc2 1000)
set -e
DATASET="${1:-tcc2}"
N="${2:-1000}"
echo "=== Pipeline de preparação — Random Projection (dataset ${DATASET}, N=${N}) ==="

# Ativa o venv do projeto (venv/, o do requirements.txt) se existir; .venv/ só como alternativa
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "[1/2] Gerando matrizes posicionais k-mer..."
python src/01_kmer_matrix.py --dataset "${DATASET}"

echo "[2/2] Projeção Random + vetores finais..."
python src/02_project_random.py --dataset "${DATASET}" --n-components "${N}"

if [ "${DATASET}" = "tcc1" ]; then PROC="data/processed"; MODELS="models"; else PROC="data/processed_v2"; MODELS="models/v2"; fi
echo ""
echo "=== Preparação concluída ==="
echo "Arrays para classificação:"
echo "  ${PROC}/X.npy        — (n_seq, ${N})"
echo "  ${PROC}/y.npy        — (n_seq,) labels 1/0"
echo "  ${PROC}/metadata.json"
echo "Vetores individuais: ${PROC}/vectors/"
echo "Modelo:              ${MODELS}/random_projection.pkl"
