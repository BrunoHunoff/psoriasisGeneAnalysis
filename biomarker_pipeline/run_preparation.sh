#!/bin/bash
# Uso: bash run_preparation.sh [tcc1|tcc2] [N] [--k K | --pattern MASCARA...]   (padrão: tcc2 1000, k=3)
#      N=0 → sem projeção. Ex.: bash run_preparation.sh tcc2 4000 --k 4
set -e
DATASET="${1:-tcc2}"
N="${2:-1000}"
shift $(( $# < 2 ? $# : 2 ))
REP_ARGS=("$@")
echo "=== Pipeline de preparação — Random Projection (dataset ${DATASET}, N=${N}, representação: ${REP_ARGS[*]:-k=3}) ==="

# Ativa o venv do projeto (venv/, o do requirements.txt) se existir; .venv/ só como alternativa
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "[1/2] Gerando matrizes posicionais k-mer..."
python src/01_kmer_matrix.py --dataset "${DATASET}" "${REP_ARGS[@]}"

echo "[2/2] Projeção Random + vetores finais..."
python src/02_project_random.py --dataset "${DATASET}" --n-components "${N}" "${REP_ARGS[@]}"

if [ "${DATASET}" = "tcc1" ]; then PROC="data/processed"; else PROC="data/processed_v2"; fi
echo ""
echo "=== Preparação concluída ==="
echo "Arrays para classificação (representações além do k=3 contínuo ficam em ${PROC}/{representação}/):"
echo "  ${PROC}/X_n{dim}.npy — (n_seq, dim); dim = N, ou o número de k-mers se N=0"
echo "  ${PROC}/y.npy        — (n_seq,) labels 1/0"
echo "  ${PROC}/metadata.json"
