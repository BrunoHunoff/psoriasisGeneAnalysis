#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== Setup e execução do pipeline de biomarcadores ==="
echo ""

# Cria o ambiente virtual se não existir
if [ ! -d ".venv" ]; then
    echo "[setup] Criando ambiente virtual Python..."
    python3 -m venv .venv
fi

# Ativa o ambiente virtual
source .venv/bin/activate

# Instala dependências se necessário
if ! python -c "import biopython" 2>/dev/null; then
    echo "[setup] Instalando dependências (pode levar alguns minutos)..."
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
    echo "[setup] Dependências instaladas."
else
    echo "[setup] Dependências já instaladas."
fi

echo ""
echo "ATENÇÃO: coloque seus arquivos FASTA em data/raw/ antes de continuar."
echo "Formatos aceitos: positive.fasta, positive.csv ou positive.txt"
echo "                  negative.fasta, negative.csv ou negative.txt"
echo ""
echo "Se a pasta data/raw/ estiver vazia, dados sintéticos serão gerados"
echo "automaticamente para fins de desenvolvimento."
echo ""
read -p "Pressione Enter para continuar ou Ctrl+C para cancelar..."

mkdir -p data/processed models reports

echo ""
echo "[1/5] Validando e organizando dados..."
python src/01_ingest.py

echo ""
echo "[2/5] Gerando matriz k-mer..."
python src/02_kmer.py

echo ""
echo "[3/5] Redução dimensional SVD..."
python src/03_svd.py

echo ""
echo "[4/5] Treinamento e avaliação dos modelos..."
python src/04_train.py

echo ""
echo "[5/5] Interpretabilidade SHAP..."
python src/05_shap.py

echo ""
echo "==================================================="
echo " Concluído! Verifique:"
echo "   reports/  → gráficos e métricas"
echo "   models/   → modelos treinados"
echo "==================================================="
