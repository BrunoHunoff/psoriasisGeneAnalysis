# Pipeline de Identificação de Biomarcadores Proteicos

TCC de Engenharia de Software — Pipeline completo de machine learning para
identificação de biomarcadores em sequências proteicas usando representação
k-mer, redução SVD e interpretabilidade SHAP.

---

## Contexto científico

**Por que proteínas?** Proteínas são os executores funcionais do genoma.
Biomarcadores proteicos (presença ou padrão de aminoácidos específicos)
têm aplicação direta em diagnóstico clínico, descoberta de alvos
terapêuticos e estudos de doenças.

**Por que k-mer?** Representar uma sequência como frequência de sub-sequências
de comprimento k (k=3 → trímeros de aminoácidos) transforma cadeias de
comprimento variável em vetores de tamanho fixo (20³ = 8.000 features),
sem depender de alinhamento global. Essa abordagem captura padrões locais
de composição que são biologicamente informativos.

**Por que SVD?** A matriz k-mer é altamente esparsa e redundante. A
decomposição em valores singulares truncada (SVD) realiza uma projeção
ortogonal — análoga a converter um objeto 3D em 2D — que preserva a
estrutura essencial enquanto reduz ruído e custo computacional.

**Por que SHAP?** SHapley Additive exPlanations fornece importâncias
de features consistentes e teoricamente fundamentadas, permitindo mapear
a contribuição de cada componente SVD de volta ao espaço k-mer original
e identificar quais trímeros de aminoácidos distinguem marcadores de
não-marcadores.

---

## Como colocar seus dados

Coloque **um arquivo por biomarcador** na pasta `data/raw/` antes de rodar.
O nome do arquivo vira o rótulo da classe automaticamente.

```
data/raw/
├── EGFR.fasta       → classe "EGFR"
├── TP53.fasta       → classe "TP53"
├── BRCA1.fasta      → classe "BRCA1"
└── ...
```

### Formatos aceitos por arquivo

| Extensão | Formato | Observações |
|----------|---------|-------------|
| `.fasta` / `.fa` | FASTA padrão | Exportado do UniProt/NCBI — recomendado |
| `.csv` | CSV | Colunas obrigatórias: `id`, `sequence` |
| `.txt` | Texto puro | Uma sequência por linha, sem header |

O pipeline precisa de **pelo menos 2 arquivos** para funcionar (mínimo 2 classes).

Se a pasta estiver vazia e `use_synthetic_if_empty: true` no `config.yaml`,
3 classes sintéticas são geradas automaticamente para desenvolvimento.

---

## Instalação

```bash
# Recomendado: ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

---

## Como executar

### Opção 1 — Pipeline completo (recomendado)

```bash
cd biomarker_pipeline
bash run_pipeline.sh
```

### Opção 2 — Módulo por módulo

```bash
python src/01_ingest.py   # Validação e organização dos dados
python src/02_kmer.py     # Geração da matriz k-mer
python src/03_svd.py      # Redução SVD + scree plot
python src/04_train.py    # Treinamento e avaliação
python src/05_shap.py     # Interpretabilidade e ranking de biomarcadores
```

Cada módulo lê seus inputs do disco e grava outputs no disco — qualquer
etapa pode ser reexecutada isoladamente sem rodar o pipeline todo.

### Notebooks

```bash
jupyter notebook notebooks/
```

| Notebook | Conteúdo |
|----------|----------|
| `01_EDA.ipynb` | Análise exploratória: distribuições, heatmap k-mer, t-SNE |
| `02_scree_plot.ipynb` | Decisão do número de componentes SVD |
| `03_results.ipynb` | Resultados finais: métricas, ROC/PRC, SHAP, biomarcadores |

---

## Arquivos gerados em `reports/`

| Arquivo | Descrição |
|---------|-----------|
| `scree_plot.png` | Variância acumulada SVD com threshold e ponto escolhido |
| `svd_summary.json` | n_components escolhido e variância capturada |
| `roc_curves.png` | Curvas ROC sobrepostas dos 3 modelos |
| `metrics.json` | Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PRC por modelo |
| `shap_summary.png` | Beeswarm plot SHAP no espaço SVD |
| `top_kmers_bar.png` | Bar plot dos top 20 k-mers por importância SHAP |
| `top_kmers.csv` | Ranking completo com sequência e importância de cada k-mer |

---

## Referências científicas

1. **Asgari, E. & Mofrad, M. R. K.** (2015). Continuous Distributed
   Representation of Biological Sequences for Deep Proteomics and Genomics.
   *PLOS ONE*, 10(11), e0141287.
   https://doi.org/10.1371/journal.pone.0141287

2. **Lundberg, S. M. & Lee, S.-I.** (2017). A unified approach to
   interpreting model predictions. *Advances in Neural Information
   Processing Systems*, 30.

3. **Hallee, L. & Khomtchouk, B. B.** (2023). Machine learning classifiers
   predict key genomic and evolutionary traits across the kingdoms of life.
   *Scientific Reports*, 13, 2088.
   https://doi.org/10.1038/s41598-023-28965-7

4. **Rives, A. et al.** (2021). Biological structure and function emerge
   from scaling unsupervised learning to 250 million protein sequences.
   *PNAS*, 118(15).
   https://doi.org/10.1073/pnas.2016239118

5. **Shen, H. B. & Chou, K. C.** (2007). PseAAC: A flexible web server
   for generating various kinds of protein pseudo amino acid composition.
   *Analytical Biochemistry*, 373(2), 386–388.
   https://doi.org/10.1016/j.ab.2007.10.012

6. **Halko, N., Martinsson, P.-G. & Tropp, J. A.** (2011). Finding
   Structure with Randomness: Probabilistic Algorithms for Constructing
   Approximate Matrix Decompositions. *SIAM Review*, 53(2), 217–288.
   https://doi.org/10.1137/090771806
