# CLAUDE.md — Contexto do projeto

TCC de Bacharelado em Engenharia de Software — Universidade Positivo (2026).

**Título:** Machine learning para classificação de proteínas associadas à psoríase a partir de k-mers de sequências de aminoácidos.

**Autores:** Igor Felipe Viana, Bruno Rafael Hunoff de Freitas, Nicolas Czekoski Yasumoto, Laura Reded de Melo.
**Orientador:** Prof. Aryel Marlus.

O texto do TCC I está no PDF `Documentação_-_TCC__ABNT_ (2).pdf`. ⚠️ A versão em Markdown (`documentacao.md`) **foi perdida em 2026-10-10** junto com outros arquivos não versionados (ver §6) — se for recuperada, volta a ser a fonte de verdade dos quadros. **Sempre que alterar resultados numéricos no código, os quadros do documento precisam ser reconferidos.**

---

## 1. O que o projeto faz

Classificação binária **de proteínas** (não de pacientes) a partir da sequência de aminoácidos, sem alinhamento:

```
FASTA → matriz posicional one-hot de k-mers (L-k+1, 8000)
      → Gaussian Random Projection (8000 → N)
      → mean pooling sobre as janelas → vetor (N,)
      → classificador supervisionado → POSITIVO (psoríase) / NEGATIVO
```

- `k = 3` → vocabulário `20³ = 8.000` k-mers (20 aminoácidos canônicos, `ACDEFGHIKLMNPQRSTVWY`).
- A projeção é ajustada **sobre a matriz global empilhada de todas as janelas** de todas as sequências, e só depois as linhas são reagrupadas por sequência para o mean pooling.
- `N` avaliado em `{300, 400, 500, 1000}`. **N=1000 é a configuração final** — é o que os modelos salvos e os scripts 06/07 assumem.
- Base metodológica: Guo et al. (2021), k-mers + classificação proteica.

### Dataset do TCC I (223 sequências, ~1:1)

| Grupo | Diretório | Label | N | Composição |
|---|---|---|---|---|
| Positivos | `data/raw/` | 1 | 111 seqs / 28 arquivos | Genes notáveis dos 21 loci de suscetibilidade confirmados por Tsoi et al. (2012) — HLA-C, HLA-B, IL12B, IL23R, IL23A, TRAF3IP2, TNFAIP3, TNIP1, TYK2, ERAP1, NFKBIA, REL, IFIH1, LCE3B/D, NOS2, ZMIZ1, RNF114, UBE2L3, STAT2, IL4, IL13, IL28RA, KCNH7, FBXL19, PRDX5, PRSS53, RPS6KA4 — mais homólogos de múltiplos organismos (máx. 10 por gene) |
| Negativos | `data/raw_negativos/` (só os 30 arquivos do commit `cc6a498`) | 0 | 112 seqs / 30 arquivos | Três níveis: **housekeeping** (ACTB, ALB, ALDOA, ATP5F1A, CAT, COL1A1, ENO1, GAPDH, HBA1, HBB, INS, LDHA, PGK1, PKM, SOD1, TPI1, TUBB, VIM — 72 seqs), **imunes não ligadas à psoríase** (CD3E, IFNG, IL2, IL5, IL10 — 20 seqs), **doenças com sobreposição genética** — Crohn (NOD2, ATG16L1, IRGM) e artrite reumatoide (CTLA4, PADI4, PTPN22, TRAF1) — 20 seqs. Máx. 4 seqs por proteína |

O label vem **do diretório de origem**, não de anotação manual. STAT4 foi excluída dos negativos por também ser locus de psoríase em Tsoi et al. (2012).

`data/input/` guarda as 5 proteínas de **validação externa do TCC I**: INS (negativa treinada), PTPN22 (negativa não treinada), IL12B (positiva treinada), CARD14 (positiva não treinada, via epitelial), RUNX3 (positiva não treinada, via imune sistêmica).

---

## 2. Estrutura e fluxo

Tudo roda a partir de `biomarker_pipeline/` como diretório de trabalho (os caminhos nos scripts são **relativos**, ex. `"data/processed"`).

```
biomarker_pipeline/
├── src/
│   ├── 00_build_gene_lists.py # TCC II: SD20 da Dand 2025 → listas de genes de treino e de teste cego
│   ├── 00_fetch_uniprot.py    # TCC II: lista de genes → FASTAs Swiss-Prot + homólogos (positivos)
│   ├── 00_fetch_negatives.py  # TCC II (Igor): coleta de negativos difíceis → data/raw_negativos/
│   ├── 01_kmer_matrix.py      # FASTA → matrices/{key}_matrix.npy + sequence_registry.json + kmer_vocab.json
│   ├── 02_project_random.py   # --n-components N → X.npy, y.npy, metadata.json, random_summary.json, models/random_projection.pkl
│   ├── 05_train_evaluate.py   # --model M → GridSearchCV 10-fold → reports/results_{M}.json + models/classifier_{M}_n{N}.pkl
│   ├── 06_train_final.py      # --model M → treina em 100% dos dados → classifier_{M}_n1000.pkl + _meta.json
│   ├── 07_predict.py          # --model M [--fasta …] → prediz FASTAs de data/input/
│   ├── run_experiments.py     # orquestra 02+05 para N ∈ {300,400,500,1000} × 6 modelos → tabelas_resultados.tsv
│   └── run_predictions.py     # roda os 6 modelos nas 5 proteínas de validação externa → tabelas_predicoes.tsv
├── data/{raw,raw_negativos,raw_v2,teste_cego_v2,gene_lists,input,processed}/
├── models/    # *.pkl gitignored; *_meta.json versionados
├── reports/   # results_{modelo}_n{N}.json, tabelas_*.tsv, experimento2/*.md, negativos/, coleta/
├── notebooks/ # 01_EDA, 02_scree_plot, 03_results — DESATUALIZADOS, ver §4
└── run_preparation.{sh,ps1}
```

**Não existem `03_*.py` nem `04_*.py`** — a numeração tem lacuna herdada de uma versão anterior da pipeline (que usava SVD e one-class classification). Não é um arquivo faltando.

`sequence_registry.json` é o contrato entre 01 e 02: a **ordem** das entradas define o empilhamento da matriz global e o reagrupamento no pooling. Não reordenar.

### Comandos

```bash
cd biomarker_pipeline
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

bash run_preparation.sh                                   # 01 + 02 (ver ressalva §4)
python src/run_experiments.py                             # experimento completo (4 N × 6 modelos)
python src/06_train_final.py --model logistic_regression  # modelo final
python src/run_predictions.py                             # validação externa em lote
```

### Modelos (`--model`)

`logistic_regression`, `svm`, `random_forest`, `gradient_boosting`, `naive_bayes`, `knn`.

LR, SVM e kNN são envolvidos em `Pipeline([StandardScaler, modelo])` — o scaler é ajustado dentro de cada fold. RF, GB e NB rodam sem scaler.

Grids em `PARAM_GRIDS` (`05_train_evaluate.py`); `refit="roc_auc"`; métricas reportadas a partir de `cv_results_` no `best_index_` — ou seja, são as métricas do melhor conjunto de hiperparâmetros, não de um refit separado.

---

## 3. Resultados consolidados (TCC I)

Melhor configuração: **N=1000, Regressão Logística (C=0.1)** — ROC-AUC 99,8% ± 0,3%, MCC 95,0% ± 5,5%, acurácia 97,4%.

Padrões estáveis em todas as configurações de N:
- LR, SVM e RF: ROC-AUC > 97%, MCC > 83%. Desempenho cresce monotonicamente de N=300 a N=1000.
- GB intermediário; kNN com sensibilidade altíssima e especificidade baixa; Naive Bayes ruim (ROC-AUC 73–78%).
- Validação externa: modelos generalizam para **RUNX3** (LR 67,6%, SVM 90,3%) mas **não** para **CARD14** (errado em LR, SVM, GB, NB).

Números por modelo × N: `reports/results_{modelo}_n{N}.json` e `reports/experimento2/experimento2_vec{N}.md`. O commit "cleanup old results" (`49b2d57`) só removeu duplicatas sem sufixo, a rodada N=2000 e relatórios da versão com 226 amostras — os definitivos estão intactos.

---

## 4. Armadilhas conhecidas (leia antes de mexer)

1. **Sem agrupamento por proteína na CV.** `05_train_evaluate.py` usa `StratifiedKFold`. O commit `647049b` usava `StratifiedGroupKFold` agrupando pelo campo `protein` do `metadata.json`; isso foi perdido no refactor para `GridSearchCV` (`cc6a498`). Com até 10 homólogos por gene, **as métricas atuais provavelmente estão otimistas**. Corrigir **antes** de qualquer fine tuning do TCC II.
2. **`N_COMPONENTS` está triplicado e inconsistente.** `02_project_random.py` tem default `500`; `06_train_final.py` e `07_predict.py` têm `1000` hard-coded. Para o fluxo N=1000, rode `python src/02_project_random.py --n-components 1000` antes de 06/07.
3. **`05_train_evaluate.py` sempre escreve `reports/results_{modelo}.json`** (sem `_n{N}`). É `run_experiments.py` que copia para o nome com sufixo.
4. **Os notebooks estão quebrados** — referenciam artefatos da pipeline antiga. Não use como fonte de verdade.
5. **`models/random_projection.pkl` é sobrescrito a cada execução de 02.** Sempre reajuste 02 → 05/06 na mesma sessão.
6. **`models/*.pkl` está no `.gitignore`**, mas alguns `.pkl` já foram commitados antes da regra.
7. **`01_kmer_matrix.py` lê `data/raw_negativos/` inteiro.** Desde o commit `8a20e30` (Igor) esse diretório tem 406 arquivos / 1.132 seqs, não os 30 / 112 do TCC I — **rodar a pipeline hoje não reproduz o TCC I**. Os arquivos novos precisam ir para `data/raw_negativos_v2/`.
8. **CLAUDE.md está no `.gitignore`** (commit `d45db7d`). Ele e outros arquivos não versionados já se perderam uma vez — faça backup ou versione o que importa.

---

## 5. Convenções de código

- Python 3.10+ (testado até 3.13). Deps fixadas em `requirements.txt`. Venv em `biomarker_pipeline/venv/`. `certifi` é necessário para HTTPS no Python do macOS (`CERTIFICATE_VERIFY_FAILED`); `openpyxl` para ler a SD20.
- **Português** em docstrings, comentários, logs e nomes de relatórios. Identificadores em inglês.
- Constantes em CAIXA_ALTA no topo de cada script; **sem arquivo de config** — parâmetros por `argparse` ou editando a constante.
- Todo script tem docstring de módulo com o "Uso:", `logging` para stdout no formato `%(asctime)s [%(levelname)s] %(message)s`, e `main()` sob `if __name__ == "__main__"`.
- `RANDOM_STATE = 42` em todo lugar.
- Erros de pré-requisito → `log.error(...)` explicando qual script rodar antes + `sys.exit(1)`.
- Saídas para o TCC em TSV para colar no Google Docs (`reports/tabelas_*.tsv`), com linha "Fonte: Elaborado pelos autores (ano)."

---

## 6. TCC II — expansão do dataset (em andamento)

**Objetivo do grupo:** expandir as proteínas mantendo a metodologia → retreinar → fine tuning → teste cego + amostragem sistemática → atualizar documento e apresentação.

**Decisão:** treinar com os loci **Known** e testar às cegas nos loci **Newly reported** de Dand et al. (2025) — validação prospectiva temporal, não holdout arbitrário. No texto, chamar de "validação externa prospectiva", não de "grupo controle".

Fonte única: **Dand et al. (2025), Nat Commun 16:2051, Supplementary Data 20** (`data/gene_lists/dand2025_supplementary_data20.xlsx`, baixado de `static-content.springer.com`, MOESM3 — o arquivo contém todas as SDs; a 20 é a aba usada). A coluna "Established psoriasis loci in Europeans" particiona as 109 regiões em `Known` (59) e `Newly reported` (50).

⚠️ **Known ≠ "os estudos do Tsoi".** Nota (a) da SD20: os loci estabelecidos são os *"65 previously reported psoriasis loci reported collectively in Tsoi et al Nat Commun 2017, Dand et al Hum Mol Genet 2017 and Patrick et al Nat Commun 2018"*. Tsoi 2017 já é cumulativo (inclui 2012 e 2015), então Known = Tsoi + loci de dois outros estudos. A SD20 não atribui loci por estudo. No texto: "loci estabelecidos segundo Dand et al. (2025)".

**RUNX3 (região 5) e CARD14 (região 93) são Known, mas ficam FORA do treino** (`TCC1_VALIDATION_HOLDOUT` em `00_build_gene_lists.py`) para repetir a validação externa do TCC I nas mesmas condições ("positiva não treinada"). Custo: 2 genes / 4 seqs. A comparação mede o efeito da v2 como um todo (dados, Mammalia, CV), não só do número de loci.

### Geração das listas — `src/00_build_gene_lists.py`

- **Treino (96 genes de 54 regiões; 98 menos RUNX3/CARD14):** coluna "Previously proposed" (nota f — Stuart, Tsoi, Hambro & Elder, 2018), o mesmo critério dos 28 genes do TCC I. Decisões editoriais: remove não codificantes (`MHC`, `HLA`, `LINC00330`, `LINC01885`, `SNORD74`, `RN7SKP9`, `RP11-61O1.1`, `LCE3D-LCE3E`); expande `LCE3C/B`, `STAT5A/B`, `RNF145 (IL12B)`; descarta os 4 genes só-PsA (`ADRA1B`, `CSF2`, `HLA-DRB1`, `P4HA2`). Regiões 78, 79, 82 sem gene codificador. `TNFSF15` incluído.
- **Teste cego (49 regiões; região 28 excluída):** a SD20 não traz "Previously proposed" para inéditos, então o gene é escolhido pela evidência mais forte — **A** PAV ou alvo terapêutico (10 regiões), **B** TWAS (4), **C** sinal dentro do gene (17), **D** gene mais próximo de sinal intergênico (18). Nível por região em `dand2025_validacao_ineditos.tsv` — reportar o teste cego estratificado por nível, D é evidência fraca. Região 28 excluída: bloco de LD adjacente à 29 (Known), TP63 já está no treino.

### Coleta — `src/00_fetch_uniprot.py`

Mesmos filtros de `00_fetch_negatives.py` para que taxonomia e tamanho não virem atalho entre classes: `reviewed:true`, `fragment:false`, **Mammalia** (`--taxon 40674`), 50–2000 aa, sem X/Z/B, humano obrigatório, uma seq por espécie, até `--max-seqs` (10, como no TCC I), desempate determinístico por accession. `--out-dir` obrigatório e recusa `data/raw`.

**Validação de símbolo:** `gene_exact:` casa sinônimos (NOS2→NANOS2, PRDX5→Prdx6, BRAP→C6orf89, MBD2→Dpep2, REL→Rela). Só é aceito nome primário igual ao pedido ou renomeação em `GENE_ALIASES` (`DDX58`→RIGI, `PTRF`→CAVIN1, `C1orf68`→KPLCE, `IL28RA`→IFNLR1, `ADC`→AZIN2). Ao adicionar genes, conferir os descartes no log.

| Diretório | Conteúdo |
|---|---|
| `data/raw/` | **TCC I, congelado** — 28 genes, 111 seqs |
| `data/raw_v2/` | **Treino TCC II** — 96 genes, 316 seqs (Mammalia) |
| `data/teste_cego_v2/` | **Teste cego** — 46 genes, 147 seqs. Fora: CHD7 e WNK1 (> 2000 aa), TMEM75 (hoje LINC02912) |
| `data/raw_negativos/` | 30 arquivos do TCC I + 376 do Igor (misturados — ver §4.7) |
| `reports/coleta/*.json` | status por gene, organismos e descartes de cada coleta |

### Negativas (Igor, `00_fetch_negatives.py`, commit `8a20e30`)

Fontes: parálogos dos 28 positivos do TCC I (família UniProt/Pfam + BLOSUM62), genes de outras doenças imunomediadas (Open Targets), citocinas/receptores. Exclusão por Open Targets, GWAS Catalog e texto UniProt. Mammalia, máx. 4/gene.

| Nível | TCC I | Novos | 
|---|---|---|
| Fácil — housekeeping | 18 genes / 72 seqs | **0** |
| Intermediário — outras doenças imunes | 7 / 20 | 191 / 534 |
| Difícil — imunes não-psoríase | 5 / 20 | 98 / 254 |
| Difícil — parálogos | — | 87 / 232 |

### Pendências

1. **Reorganizar negativas:** mover os 376 arquivos do Igor para `data/raw_negativos_v2/`; **remover `STAT5A_paralogo_STAT2_dificil.fasta`** (STAT5A é positivo de treino, região 91 — o filtro do Igor só compara com os 28 genes do TCC I).
2. **Amostragem sistemática:** proporção-alvo por nível (decisão do grupo: ⅓ cada ou a do TCC I) e amostra estratificada até 1:1 com as 316 positivas; **separar genes negativos para o teste cego** (sem negativas o teste cego só mede sensibilidade). Coletar mais housekeeping (nível vazio).
3. **CV agrupada por gene** (§4.1) antes do fine tuning.
4. **Homólogos byte-idênticos em `raw_v2`:** 7 grupos (CFL1 ×4, FASLG ×3, PSMA6 ×3, UBE2L3 ×3, IL4/KLRK1/PTEN ×2). Inofensivos com CV agrupada por gene; deduplicar se a CV não for agrupada.
5. **Erros herdados no TCC I (`data/raw/`, congelado):** `PRDX5.fasta` tem `P83200` (fragmento de 16 aa com X); `NOS2.fasta` tem 2 seqs de NANOS2, `PRDX5.fasta` tem Prdx6, `FBXL19.fasta` tem FBXL19-AS1; `TNIP1.fasta` tem uma vírgula solta; `data/raw_negativos/HBB_housekeeping_facil.fasta` contém X/Z/B. Vale nota de limitação no documento.
6. **`PTPN22` não é "negativa não treinada".** O TCC I descreve assim a PTPN22 de `data/input/`, mas `PTPN22_Artrite_Reumatoide.fasta` está nos negativos de treino desde o commit `cc6a498`. Conferir a seq e corrigir o texto ou o arquivo.

### Arquivos perdidos em 2026-10-10

Por volta do `git pull --rebase` de 2026-10-10 15:29, sumiram todos os não versionados: `CLAUDE.md`, `documentacao.md`, `src/00_fetch_uniprot.py`, `data/gene_lists/*`, `data/raw_v2/*` e alterações locais de `README.md`/`requirements.txt`. Causa não identificada. Recriados em 2026-10-11: este arquivo, as listas, os scripts 00 e os FASTAs v2. `documentacao.md` **não** foi recuperado.

---

## 7. TCC II — outras direções previstas

Registradas em "Trabalhos Futuros" (§6.1 do documento):

1. **Variar k** — k=2 (400 dims) e k=4 (160.000 dims) contra k=3.
2. **Refinar modelos** — grids mais amplos; redes neurais.
3. **Aplicar ao proteoma humano completo** (~20.000 proteínas) para triagem de candidatos.

Corrigir o vazamento por homólogos (§4.1) deve preceder qualquer conclusão nova.
