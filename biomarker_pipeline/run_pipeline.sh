#!/bin/bash
set -e

echo "=== Pipeline de identificação de biomarcadores ==="
echo ""
echo "ATENÇÃO: coloque seus arquivos FASTA em data/raw/ antes de continuar."
echo "Formatos aceitos: positive.fasta, positive.csv ou positive.txt"
echo "                  negative.fasta, negative.csv ou negative.txt"
echo ""
echo "Se a pasta data/raw/ estiver vazia, dados sintéticos serão gerados"
echo "automaticamente (configurável em config.yaml → use_synthetic_if_empty)."
echo ""
read -p "Pressione Enter para continuar ou Ctrl+C para cancelar..."

# Garante que os diretórios de saída existem
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
echo "   reports/  → scree_plot, roc_curves, shap_summary, top_kmers.csv"
echo "   models/   → best_model.pkl e modelos individuais"
echo "==================================================="
