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
- No TCC I a projeção era ajustada **sobre a matriz global empilhada de todas as janelas** e só depois as linhas eram reagrupadas por sequência para o mean pooling. Desde 2026-10-20 cada sequência é projetada separadamente, com a mesma matriz gaussiana — resultado idêntico bit a bit (ver §2, "Projeção").
- `N` avaliado em `{300, 400, 500, 1000}` no TCC I. **N=1000 é a configuração final do TCC I** e o padrão de `--n-components` em todos os scripts.
- TCC II: k e N viram parâmetros (`--k`, `--pattern`, `--n-components`, 0 = sem projeção) para a grade de representação — ver §2 e §6. **Configuração escolhida para o TCC II: k=3 contínuo, N=4000** (decisão de 2026-10-29, §6). O padrão dos scripts continua 1000 — passar `--n-components 4000` explicitamente.
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
│   ├── common.py              # DATASETS, representação k-mer (--k/--pattern), caminhos, vetorização e projeção compartilhadas
│   ├── 01_kmer_matrix.py      # FASTA → matrices/{key}_matrix.npz (esparsa) + sequence_registry.json + kmer_vocab.json
│   ├── 02_project_random.py   # --n-components N (padrão 1000; 0 = sem projeção) → X_n{dim}.npy, y.npy, metadata.json, random_summary_n{dim}.json
│   ├── 05_train_evaluate.py   # --model M → GridSearchCV 10-fold → results_{M}_n{dim}.json + classifier_{M}_n{dim}.pkl
│   ├── 06_train_final.py      # --model M → treina em 100% com os best_params do 05 → classifier_{M}_n{dim}.pkl + _meta.json
│   ├── 07_predict.py          # --model M [--fasta …] → prediz FASTAs de data/input/
│   ├── 08_blind_test.py       # TCC II: --tag {antes|otimizado_misturado|otimizado_por_gene} --n-components N (obrigatório) --model M|all → teste cego (UMA vez por braço) → reports/v2/teste_cego_{tag}_*.json + tabelas_teste_cego_{tag}.tsv
│   ├── 09_optimize_hyperparams.py # TCC II, braços A/B: --arm {misturado|por_gene} --model {logistic_regression|svm|trees|boosting|mlp|comite|all} --n-components 4000 [--piloto N] → Optuna (TPE), CV 5×10 → reports/v2/otimizacao/{braço}/ + models/v2/otimizacao/{braço}/
│   ├── run_experiments.py     # orquestra 01+02+05 para uma representação × --n-values × --models → tabelas_resultados.tsv + tabelas_representacoes.tsv
│   ├── run_predictions.py     # roda os 6 modelos nas 5 proteínas de validação externa → tabelas_predicoes.tsv
│   └── relatorios/            # geradores dos PDFs de reports/v2/ (reportlab; fontes Arial do macOS)
├── data/{raw,raw_negativos,raw_v2,raw_negativos_v2,teste_cego_v2,teste_cego_negativos_v2,gene_lists,input,processed,processed_v2}/
├── models/    # TCC I; models/v2/ = TCC II. *.pkl gitignored; *_meta.json versionados
├── reports/   # TCC I: results_{modelo}_n{N}.json, tabelas_*.tsv, experimento2/; reports/v2/ = TCC II; negativos/, coleta/
├── docs/      # plano_otimizacao.md — PRÉ-REGISTRO do experimento final (três braços); ler antes de rodar o teste cego
├── notebooks/ # 01_EDA, 02_scree_plot, 03_results — DESATUALIZADOS, ver §4
└── run_preparation.{sh,ps1}
```

**Não existem `03_*.py` nem `04_*.py`** — a numeração tem lacuna herdada de uma versão anterior da pipeline (que usava SVD e one-class classification). Não é um arquivo faltando.

**Datasets (`--dataset`, padrão `tcc2`, definidos em `common.DATASETS`):** `tcc1` lê `data/raw` + `data/raw_negativos` e grava em `data/processed`, `models/`, `reports/` (os caminhos dos quadros do TCC I); `tcc2` lê `data/raw_v2` + `data/raw_negativos_v2` e grava em `data/processed_v2`, `models/v2/`, `reports/v2/`. Um nunca sobrescreve o outro.

**Representação (`--k`, `--pattern`, `--n-components`; em 01, 02, 05, 06, 07, 08, run_experiments e run_predictions):** cada padrão é uma máscara sobre a janela — `111` é o k=3 contínuo, `1101` lê as posições 1, 2 e 4 ("k=3 pulando uma posição", sugestão do orientador); `--pattern 1101 1011` combina os dois lado a lado (16.000 colunas, a linha tem um 1 por padrão; máscaras combinadas precisam ter o mesmo k e o mesmo comprimento). Nomes: `k3`, `k2`, `k4`, `k3_1101`, `k3_1101+1011`. **k3 contínuo usa os diretórios do dataset**; as outras gravam em subdiretório: `data/processed_v2/k4/`, `models/v2/k4/`, `reports/v2/k4/`. `--n-components 0` = sem projeção (vetor = frequência relativa de cada k-mer); os arquivos levam a **dimensão final** no nome, então "sem projeção" é `n8000` (k3), `n400` (k2), `n160000` (k4). N ≥ número de k-mers é recusado (use 0).

**Matrizes esparsas:** desde 2026-10-20 o 01 salva as matrizes one-hot em CSR (`.npz`). A densa do TCC II teria ~325 mil janelas × 8.000 (~10 GB).

**Projeção (`common.kmer_vectors`):** não há mais `GaussianRandomProjection` nem `random_projection.pkl`. A matriz gaussiana é gerada pelo mesmo `RandomState(42).normal(0, 1/√N)` do sklearn, convertida para float32, em blocos de ~512 MB (k=3 cabe num bloco; k=4 com N=10.000 usa 12), e cada sequência é projetada e pooled separadamente. Verificado em 2026-10-20: `X_n1000.npy` **idêntico bit a bit** ao `X.npy` antigo do TCC II, e o 05 reproduz exatamente métricas e best_params dos `results_*_n300.json` commitados. 07/08/run_predictions **refazem** a mesma projeção a partir de (representação, N, seed) — vetor de predição == vetor de treino, verificado inclusive no caminho em blocos. Custo: k=4 N=10.000 leva 16 s e 3,3 GB de pico; k=4 sem projeção gera `X_n160000.npy` de 386 MB. Cuidado: a média do numpy muda de ordem de soma num bloco de largura 1 — `_projection_blocks` evita isso.

`sequence_registry.json` é o contrato entre 01 e 02: a **ordem** das entradas define a ordem das linhas de X, y e metadata. Não reordenar.

### Comandos

```bash
cd biomarker_pipeline
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

bash run_preparation.sh tcc2 1000 --k 4                                  # só 01 + 02 (args: dataset, N, representação)
python src/run_experiments.py --dataset tcc2                             # 01 + 02 + 05: k3, N ∈ {300,400,500,1000} × 6 modelos
python src/run_experiments.py --dataset tcc2 --k 4 --n-values 1000 4000 0 --models svm logistic_regression
python src/run_experiments.py --dataset tcc2 --resumo                    # só refaz as tabelas a partir dos JSONs
python src/06_train_final.py --model svm --dataset tcc2 --k 4 --n-components 4000   # modelo final (best_params do 05)
python src/run_predictions.py --dataset tcc2 --k 4 --n-components 4000   # 5 proteínas da validação do TCC I
python src/08_blind_test.py --tag antes --model all --n-components 4000            # teste cego — UMA vez por braço (docs/plano_otimizacao.md)
python src/09_optimize_hyperparams.py --arm por_gene --model all --n-components 4000 # braço B (A: --arm misturado); retomável
python src/08_blind_test.py --tag otimizado_por_gene --model all --n-components 4000
python src/relatorios/melhor_k_n_por_modelo.py                           # PDF da melhor representação por modelo
```

As tabelas de `run_experiments` são refeitas a partir de **todos** os `results_*_n*.json` existentes (não só os da execução): `tabelas_resultados.tsv` por representação e `reports/v2/tabelas_representacoes.tsv` com ROC-AUC e MCC de todas as representações × N (para escolher k e N).

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

1. **CV sem agrupamento por gene — decisão consciente.** `05_train_evaluate.py` usa `StratifiedKFold`: homólogos do mesmo gene podem cair em treino e teste do mesmo fold, então a CV mede "reconhece proteínas/homólogos do treino". O orientador considerou desnecessário agrupar; o grupo decidiu manter (2026-10-15). A generalização para **proteínas novas** é medida pelo teste cego (`08_blind_test.py`), separado por gene por construção. No texto, não apresentar a métrica da CV como generalização. (O commit `647049b` tinha `StratifiedGroupKFold`, se um dia quiserem a comparação.)
2. ~~`N_COMPONENTS` triplicado~~ — resolvido em 2026-10-20: todos os scripts recebem `--n-components` (padrão 1000) e a representação; nada mais é inferido de estado salvo.
3. ~~05 escrevia `results_{modelo}.json` sem sufixo~~ — desde 2026-10-20 escreve `results_{modelo}_n{dim}.json` direto. Os `results_{modelo}.json` antigos que sobraram em `reports/v2/` são cópias velhas (gitignored), ignore.
4. **Os notebooks estão quebrados** — referenciam artefatos da pipeline antiga. Não use como fonte de verdade.
5. ~~`random_projection.pkl` sobrescrito a cada 02~~ — desde 2026-10-20 não existe mais projetor salvo (a projeção é refeita de representação + N + seed) e X é por N (`X_n{dim}.npy`). Sobram arquivos antigos sem uso: `data/processed_v2/X.npy`, `random_summary.json`, `vectors/` e `models/v2/random_projection.pkl` (todos gitignored). **Representação e N precisam ser passados iguais** em 02 → 05 → 06 → 07/08; o 08 confere contra o `_meta.json` do classificador.
6. **`models/*.pkl` está no `.gitignore`**, mas alguns `.pkl` já foram commitados antes da regra.
7. **`00_fetch_negatives.py` (Igor) ainda escreve em `data/raw_negativos/`** — rodá-lo de novo volta a misturar com o TCC I; os 376 arquivos dele foram movidos para `data/raw_negativos_igor/`.
8. **CLAUDE.md está no `.gitignore`** (commit `d45db7d`). Ele e outros arquivos não versionados já se perderam uma vez — faça backup ou versione o que importa.
9. **`06_train_final.py` ignorava o fine tuning (corrigido em 2026-10-20).** Treinava com hiperparâmetros fixos (LR C=1, kNN k=5…) e salvava com o mesmo nome do modelo ajustado do 05, sobrescrevendo-o. Agora lê `best_params` de `results_{modelo}_n{N}.json`. Os números publicados no TCC I vieram dos modelos ajustados (o sandbox reproduziu RUNX3 67,6% LR / 90,3% SVM exatos).
10. **Teste cego: rodar uma vez por braço.** `08_blind_test.py --tag {antes|otimizado_misturado|otimizado_por_gene}` (obrigatório) recusa sobrescrever `reports/v2/teste_cego_{tag}_{modelo}.json` sem `--force`, e `--n-components` é **obrigatório** no 08 (o padrão 1000 dos outros scripts gastaria o teste com modelos N=1000, que existem com `_meta.json`). Escolha de modelo e hiperparâmetros só pela CV. Rodar o teste três vezes só vale porque o plano completo (`biomarker_pipeline/docs/plano_otimizacao.md`) foi commitado **antes** do braço 0, e porque os três braços são todos reportados. Para testar o script sem tocar no conjunto cego, aponte `blind_pos_dir`/`blind_neg_dir`/`reports_dir` de `common.DATASETS["tcc2"]` para cópias de FASTAs de treino num diretório temporário.

---

## 5. Convenções de código

- Python 3.11+ (venv em 3.13.7). Deps fixadas em `requirements.txt` — **alinhado em 2026-10-29 às versões do venv que geraram todos os resultados** (scikit-learn 1.8.0, numpy 2.4.6, scipy 1.17.1; antes listava 1.5.2/2.2.6, que nunca foram as usadas). Instalar exatamente essas versões em qualquer outra máquina (VM). Venv em `biomarker_pipeline/venv/`. `certifi` é necessário para HTTPS no Python do macOS (`CERTIFICATE_VERIFY_FAILED`); `openpyxl` para ler a SD20.
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
| `data/raw_negativos/` | **TCC I, congelado** — 30 arquivos, 112 seqs (idêntico a `cc6a498`) |
| `data/raw_negativos_igor/` | 376 arquivos originais do `00_fetch_negatives.py` — só referência; substituídos pelo pool recoletado |
| `data/negativos_pool_v2/` | Pool de negativos recoletado com os critérios das positivas — 416 genes, 1.159 seqs |
| `data/raw_negativos_v2/` | **Negativos de treino** — 111 genes, 316 seqs (106 fácil / 105 interm. / 105 difícil) |
| `data/teste_cego_negativos_v2/` | **Negativos do teste cego** — 56 genes, 147 seqs (49 / 49 / 49) |
| `reports/coleta/*.json` | status por gene, organismos e descartes de cada coleta |

### Negativas — candidatos, coleta e amostragem

Fluxo: `00_build_negative_candidates.py` → `00_fetch_uniprot.py --max-seqs 4` (mesmos critérios das positivas — o Igor não filtrava fragmentos nem X/Z/B) → `00_sample_negatives.py`.

- **Níveis:** fácil = housekeeping (TCC I + 60 sorteados com `RANDOM_STATE` de Eisenberg & Levanon 2013, `data/gene_lists/eisenberg2013_housekeeping.txt`); intermediário = outras doenças imunomediadas (TCC I + Open Targets do Igor); difícil = imunes não-psoríase + parálogos (Igor).
- **Exclusões:** positivos de qualquer conjunto; **todo gene codificador a <250 kb de qualquer das 109 regiões da Dand** (Known e Newly); `reports/negativos/psoriase_exclusao.tsv` do Igor; PTPN22. Lista em `data/gene_lists/negativos_excluidos_v2.tsv`. Isso derruba **19 das 30 negativas do TCC I**, incluindo INS (Open Targets genetic_association 0,04), IL2/IL5/IL10 (<250 kb de loci da Dand), NOD2, CTLA4. STAT5A também cai (positivo).
- **Amostragem:** 1:1 em sequências com os positivos de cada conjunto; ⅓ por nível; por **gene** (homólogos nunca se dividem entre treino e teste cego); subcategorias em rodízio dentro do nível; o último gene de cada nível pode entrar com parte das seqs para fechar a conta. Relatório: `reports/coleta/amostragem_negativos_v2.tsv`.
- 12 candidatos sem seq humana válida: símbolos de 2013 renomeados (APOPT1→COA8, FUK→FCSK, H2AFY→MACROH2A1, …), HBA1 (nome primário "HBA1; HBA2"), HCFC1 (> 2000 aa). Há folga no pool, não foram tratados.

### Pendências

1. ~~Reorganizar e amostrar negativas~~ — feito (ver acima).
2. **INS na comparação com o TCC I:** INS era a "negativa treinada" da validação externa, mas os critérios v2 a excluem dos negativos — na v2 ela vira negativa não treinada. Decidir se força a inclusão ou se reporta a mudança.
3. ~~CV agrupada por gene~~ — decidido manter `StratifiedKFold` (§4.1).
7. **Pipeline do TCC II:** linha de base rodada em 2026-10-20 (ver Resultados); código da grade de representação pronto em 2026-10-20 (§2). ~~(1) grade de representação~~ rodada em 2026-10-20; ~~(2) escolher k e N~~ k=3, N=4000 em 2026-10-29 (ver "Decisão de representação" abaixo). ~~(3) registrar o plano~~ — **experimento final definido em 2026-10-29: três braços, pré-registrados em `biomarker_pipeline/docs/plano_otimizacao.md`** (ver "Experimento final" abaixo). ~~braço 0~~ rodado em 2026-10-29 (ver "Resultado do braço 0" abaixo). ~~implementar a otimização~~ feito em 2026-10-29 (`src/09_optimize_hyperparams.py`, testado localmente com orçamentos mínimos: fluxo completo, regra de borda, reprodutibilidade, e o 08 dá resultado idêntico ao commitado no braço 0). ~~piloto na VM~~ feito em 2026-10-29 (21 min, ~US$ 1; ver registro de desvios do plano: rodada completa ≈ 13–17 h, US$ 26–33, pior caso US$ 78). ~~braço B~~ concluído e com teste cego em 2026-10-30 (ver "Resultado do braço B"). Falta, na ordem do plano: braço A na VM (iniciado em 2026-10-30 20:55 UTC; desligamento diário da VM às 18:00 UTC) → rodada completa → teste cego de cada braço → documento e apresentação.
4. ~~Sequências idênticas no treino~~ — **decidido manter** (orientador, reafirmado em 2026-10-29: mostram que o modelo reconhece o que já viu). Não são erro de coleta: são entradas UniProt de **espécies diferentes** com proteína 100% idêntica (ortólogos muito conservados — ex. UBE2L3 de humano, camundongo e boi). Positivos: 7 grupos, 12 seqs redundantes (CFL1, FASLG, IL4, KLRK1, PSMA6, PTEN, UBE2L3); negativos: 6 grupos, 9 (ACOT13, PDCD5, RTRAF, SNRPG, ST8SIA4, UBE2D2) — 21 de 632. Não deduplicar; se citar no texto, dizer que são ortólogos idênticos.
5. **Erros herdados no TCC I (`data/raw/`, congelado):** `PRDX5.fasta` tem `P83200` (fragmento de 16 aa com X); `NOS2.fasta` tem 2 seqs de NANOS2, `PRDX5.fasta` tem Prdx6, `FBXL19.fasta` tem FBXL19-AS1; `TNIP1.fasta` tem uma vírgula solta; `data/raw_negativos/HBB_housekeeping_facil.fasta` contém X/Z/B. Vale nota de limitação no documento.
6. **`PTPN22` não é "negativa não treinada".** O TCC I descreve assim a PTPN22 de `data/input/`, mas `PTPN22_Artrite_Reumatoide.fasta` está nos negativos de treino desde o commit `cc6a498`. Conferir a seq e corrigir o texto ou o arquivo.

### Resultados TCC II — linha de base (2026-10-20, mesmos grids do TCC I)

632 seqs (316/316), N ∈ {300,400,500,1000} × 6 modelos, 10-fold `StratifiedKFold`. Arquivos em `reports/v2/`.

| N=1000 | ROC-AUC v2 | (TCC I) | MCC v2 | (TCC I) | Melhores parâmetros |
|---|---|---|---|---|---|
| SVM | 99,6 ± 0,3 | 99,4 | 93,1 | 90,7 | C=10, gamma=scale |
| Regressão Logística | 98,4 ± 1,2 | 99,8 | 85,7 | 95,0 | C=0,1 |
| Gradient Boosting | 97,8 ± 1,5 | 95,5 | 82,1 | 74,9 | lr=0,2, 200 estágios |
| Random Forest | 97,7 ± 1,0 | 99,3 | 83,2 | 90,6 | 200 árvores, profundidade livre |
| kNN | 94,3 ± 1,7 | 89,5 | 68,7 | 67,5 | k=3 |
| Naive Bayes | 90,7 ± 2,8 | 78,1 | 64,7 | 47,0 | var_smoothing=1e-9 |

- SVM passa a ser o melhor; LR e RF caem (negativas mais difíceis); NB e kNN melhoram. kNN inverteu o perfil: agora especificidade alta (97%) e sensibilidade baixa (68%). N=1000 continua o melhor N para quase todos.
- **Vários vencedores na borda do grid** (SVM C=10, RF/GB 200, GB lr=0,2, NB 1e-9, kNN k=3) → é aí que o fine tuning deve ampliar.
- **Validação externa (5 proteínas, `reports/v2/tabelas_predicoes.tsv`) piorou nas negativas:** INS e PTPN22, agora realmente fora do treino, são chamadas de positivas por LR/SVM/RF/GB. No TCC I as duas estavam no treino (INS declarada; PTPN22 por engano, ver Pendências 6), então o acerto de antes era memorização. PTPN22 tem associação genética com psoríase no Open Targets (score 0,45) — foi por isso que os critérios v2 a excluíram das negativas; o rótulo "negativa" dela é discutível. RUNX3 passou a errar em LR/SVM; CARD14 acerta em LR (68%). n=5: anedótico — o teste cego é a medida de verdade.
- Contraste CV alta × proteínas não vistas erradas é coerente com a CV medir reconhecimento de homólogos (§4.1).

### Grade de representação (2026-10-20, mesmos grids e CV da linha de base)

128 resultados: k=2 sem projeção; k=3 N 300–4000 e sem projeção; k=3 espaçado `1101`, `1011` e combinado com N 1000, 4000 e sem projeção; k=4 N 1000–10000 e sem projeção (só SVM e LR). Resumo em `reports/v2/tabelas_representacoes.tsv`; detalhes em `reports/v2/{representação}/`.

| MCC (%) | SVM | LR | RF | GB | NB |
|---|---|---|---|---|---|
| k3, N=1000 (linha de base) | 93,1 ± 5,4 | 85,7 ± 8,4 | 83,2 ± 2,2 | 82,1 ± 6,3 | 64,7 ± 7,5 |
| k3, N=2000 | 95,0 ± 3,5 | 91,2 ± 4,4 | 87,4 ± 4,6 | 87,6 ± 6,1 | 68,8 ± 7,7 |
| k3, N=4000 | 95,3 ± 3,8 | 94,7 ± 3,2 | 85,5 ± 4,1 | 87,5 ± 6,6 | 68,3 ± 6,6 |
| k3, sem projeção (8000) | 92,1 ± 4,4 | 94,4 ± 3,9 | 94,7 ± 5,6 | 89,1 ± 3,3 | 85,8 ± 6,7 |
| k3 `1011`, sem projeção | 91,0 ± 3,6 | 94,4 ± 2,3 | **96,9 ± 3,4** | 90,7 ± 3,8 | 85,0 ± 4,1 |
| k3 `1101+1011`, sem projeção (16000) | 89,5 ± 3,3 | 93,7 ± 4,9 | 94,0 ± 3,3 | 92,5 ± 4,7 | 91,5 ± 3,8 |
| k2, sem projeção (400) | 91,9 ± 8,2 | 79,4 ± 7,7 | 92,5 ± 5,5 | 86,2 ± 6,3 | 57,3 ± 7,0 |
| k4, N=10000 | 92,3 ± 3,5 | 92,6 ± 3,9 | 83,6 ± 4,6 | 82,2 ± 5,2 | 73,2 ± 11,0 |
| k4, sem projeção (160000) | 86,0 ± 4,1 | 95,1 ± 3,7 | — | — | — |

- **ROC-AUC saturou:** as melhores configurações ficam em 99,5–99,9%, com diferenças menores que o desvio entre folds. O MCC discrimina mais, mas os desvios (3–5 pontos) também se sobrepõem entre as melhores. Pela regra de parada do plano, nenhuma representação supera **com folga** o k=3; a CV mede reconhecimento de homólogos (§4.1), então o teste cego é que decide.
- **Padrões consistentes:** (a) N maior ajuda LR e SVM até ~2000–4000, coerente com Johnson-Lindenstrauss; (b) **sem projeção, os modelos de árvore e o NB sobem muito** (RF 83 → 95, NB 65 → 86) — a projeção mistura os k-mers e atrapalha splits por feature; (c) **k=4 não supera k=3** no mesmo N; (d) k-mers espaçados não superam o contínuo de forma consistente; (e) k=2 com só 400 dimensões chega perto do k=3 N=1000 em SVM e RF.
- **Bordas do grid** (para o fine tuning): kNN k=3 (21 de 21 configurações), RF 200 árvores (21), GB lr=0,2 e 200 estágios (20), NB var_smoothing=1e-9 (16), LR C=0,01 (7, nas dimensões altas).
- **Custo:** GB é o gargalo (k3 sem projeção ~15 min; grade toda ~2h15 em 15 núcleos). **k=4 sem projeção com os 15 workers esgotou o swap** (o SVC converte 570 × 160 mil para float64, ~1–2,5 GB por worker) — rodar com `LOKY_MAX_CPU_COUNT=6` (11 min).

### Decisão de representação (2026-10-29): k=3 contínuo, N=4000

Um k e um N únicos para todos os modelos, como no TCC I (o melhor por modelo foi considerado e descartado: mais seleção sobre a CV enviesada e metodologia menos comparável). Ranking médio dos 4 modelos principais: k3 N=4000 empata em 1º com k3 sem projeção (4,2); k3 N=1000 fica em 11,0.

| MCC (%), k=3 | N=1000 | N=2000 | N=4000 | sem projeção |
|---|---|---|---|---|
| SVM | 93,1 | 95,0 | **95,3** | 92,1 |
| LR | 85,7 | 91,2 | **94,7** | 94,4 |
| GB | 82,1 | 87,6 | 87,5 | 89,1 |
| kNN | 68,7 | 68,5 | 68,1 | 68,2 |
| RF | 83,2 | 87,4 | 85,5 | **94,7** |
| NB | 64,7 | 68,8 | 68,3 | **85,8** |

**Como sustentar (defesa):** o vetor sem projeção entra no texto como **condição de controle / teto**, não é escondido.
1. **Sem perda nos modelos que usam a projeção:** em SVM, LR, GB e kNN o vetor completo não ganha além do desvio entre folds; o SVM (melhor modelo) tem com N=4000 o melhor MCC de toda a grade. Mesma informação com metade das dimensões.
2. **N escolhido por saturação, com regra definida antes** (parar quando o ganho < desvio entre folds): de 2000 → 4000 só a LR ainda ganha (+3,4 vs dp 3,2); de 4000 → sem projeção, nenhum modelo baseado na projeção ganha. Johnson-Lindenstrauss para 632 seqs: distorção ≤ 25% (N=1000), ≤ 17% (2000), ≤ 12% (4000) — explica a melhora sobre o TCC I e a estabilização.
3. **Escalabilidade:** sem projeção a dimensão é 20^k (400 / 8.000 / 160.000 / 3,2 mi); com projeção é sempre N. Evidência da própria grade: k=4 sem projeção esgotou a memória com paralelismo total e RF/GB nem foram rodados nele.

**Assumir:** a diferença 4000 × 8000 está dentro do ruído — o argumento é "tão bom quanto, com metade das dimensões e escalável", não "melhor". **RF e NB perdem com a projeção** (−9 e −17 pontos): avaliam uma variável por vez, e a projeção espalha cada k-mer por todos os componentes — interação modelo × representação, reportar e explicar. Perda de interpretabilidade (componentes não são k-mers) vai como limitação. Alternativa considerada: N=2000 (redução 4×, igual a 4000 em tudo menos LR, −3,5).

**Trabalhos futuros — k maior com dataset maior.** A projeção resolve o problema **computacional** de k maior; só mais dados resolvem o **estatístico** (k-mers exclusivos de um gene viram impressão digital → memorização). Fração dos k-mers observados presentes em ≥ 2 genes, subamostrando genes do treino (2026-10-29):

| Genes | k=3 | k=4 | k=5 |
|---|---|---|---|
| 51 (≈ TCC I) | 81% | 20% | 3% |
| 103 | 94% | 33% | 4% |
| 155 | 96% | 42% | 6% |
| 207 (TCC II) | 98% | 49% | 8% |

N=4000 mantém distorção ≤ 15% até 20 mil proteínas (JL é logarítmico em n e independente de k). Ressalvas: a projeção não cria informação (preserva também a geometria de "cada proteína isolada"); "dataset maior" = mais **genes** distintos, não mais homólogos, e os positivos só crescem com novos GWAS (21 loci em Tsoi 2012 → 109 em Dand 2025). Caminho concreto: após o teste cego, os genes dos loci inéditos podem entrar no treino de um modelo final. O código já gera a projeção em blocos — k=5 com N=4000 é viável em princípio (não testado).

### Experimento final (2026-10-29) — três braços, pré-registrados

**Fonte de verdade: `biomarker_pipeline/docs/plano_otimizacao.md`** — espaços de busca, orçamento, validações, limiar, comitê, regra de reporte e registro de desvios. O commit que o introduziu é o pré-registro; depois do braço 0 o plano não muda (desvios só no registro do próprio arquivo, nunca motivados pelo teste cego). Resumo:

| Braço | Tag no 08 | Hiperparâmetros |
|---|---|---|
| 0 — linha de base | `antes` | grids do 05, `StratifiedKFold` 10, refit ROC-AUC (`results_*_n4000.json` existentes); LR, SVM, RF, GB, NB, kNN |
| A — otimizado, genes misturados | `otimizado_misturado` | busca bayesiana (Optuna/TPE), `RepeatedStratifiedKFold` 5×10, objetivo MCC |
| B — otimizado, separado por gene | `otimizado_por_gene` | a mesma busca com `StratifiedGroupKFold` (grupo = gene) 5×10 |

- A e B só diferem no divisor de folds. Modelos otimizados: LR (L1/L2/elastic net), SVM (RBF/linear), árvores (RF/Extra Trees), boosting (GB/HistGB), MLP, e um comitê (média das probabilidades). NB e kNN só no braço 0. Limiar de decisão ajustado pelo MCC fora do fold; teste cego de A e B reportado com o limiar ajustado e com 0,5.
- **Métrica principal do TCC II: MCC** (ROC-AUC secundária — saturou). **Modelo principal de cada braço:** maior MCC médio na validação do próprio braço, desempate ROC-AUC (regra aprovada em 2026-10-29; no braço 0 dá SVM, 95,3).
- **Regra de reporte:** os três braços e todos os modelos são reportados; a comparação entre braços é o resultado. Não apresentar o melhor braço/modelo escolhido depois do teste cego como estimativa não enviesada.
- **Implementação (`09_optimize_hyperparams.py`):** estudos do Optuna em SQLite (`optuna_{modelo}.db`, gitignored, retomáveis — rodar o mesmo comando continua de onde parou; a retomada reinicia o sorteador, então uma busca interrompida não é idêntica a uma contínua). Histórico versionável em `tentativas_{modelo}.csv`; probabilidades fora do fold em `oof_{modelo}_n4000.npz` (o comitê é calculado a partir delas, sem retreinar). Grupo = gene (nome do FASTA; negativos até o primeiro `_`): 207 genes, zero genes compartilhados entre treino e validação nas 50 divisões do braço B (2.620 no A). Busca reprodutível (mesmos valores em duas execuções). Desvios de implementação registrados no próprio plano (LR via `l1_ratio`; passo da regra de borda em faixas lineares; definição do MCC do modelo principal; piloto separado).
- **VM da rodada:** `tcc-otimizacao-vm` (grupo `tcc-otimizacao`, East US, IP público estático no momento da criação, SSH `tcc@<ip>` com `~/.ssh/id_ed25519`, liberado só para o IP do Bruno — se o IP mudar, atualizar a regra `ssh-somente-ip-do-bruno` do NSG). Ambiente: Ubuntu 24.04, `uv` com Python 3.13.7 e o `requirements.txt` — versões idênticas ao Mac; `X_n4000.npy`, `y.npy` e `metadata.json` gerados na VM saíram **idênticos bit a bit** aos do Mac. Repositório clonado raso do GitHub (público). Piloto (5 tentativas por modelo, braço B): LR 42 s, SVM 6 s, árvores 27 s, boosting 161 s, MLP 7 s por tentativa.
- **Custo por tentativa (Mac, 15 núcleos, testes):** LR com L1/saga até ~4 min (C intermediário converge devagar); SVM ~15 s; MLP ~10 s; árvores e boosting dependem muito do sorteio (boosting com 2000 estágios e profundidade 6 é o pior caso).
- O braço B **não** muda a decisão de §4.1 (a CV reportada da linha de base continua `StratifiedKFold`): a separação por gene entra só como critério de escolha dos hiperparâmetros do braço B.
- **Terminologia:** "otimização de hiperparâmetros", não "fine tuning" (fine tuning = continuar o treino de um modelo pré-treinado; aqui a representação é fixa e os classificadores são treinados do zero — não há o que ajustar). Defesa: representação fixa e não aprendida é escolha de projeto; fine tuning de verdade (ESM-2, ProtBERT) fica como trabalho futuro/comparação.
- **Infraestrutura:** Azure, assinatura "Azure subscription 1" (Sponsorship, conta brunobateraf@gmail.com — a do Foundry/TeighAI; limite de gasto **desligado**, conferir saldo de créditos). East US, `Standard_F32als_v7` (32 núcleos físicos AMD, 64 GB, US$ 1,94/h), grupo de recursos `tcc-otimizacao`, apagado ao fim. Cota: 65 vCPUs regionais (16 já usados pela VM `teigh-agent`, de outro projeto — não mexer); Fasv6/FSv2 indisponíveis na assinatura; spot com cota de 3 vCPUs. Custo medido no Mac (2026-10-29): GB 3–6 min de CPU por treino de 1000 árvores, RF segundos a ~3 min, SVM/LR < 1 s. Estimativa preliminar US$ 100–200 para A+B; o piloto refina antes da rodada completa.

### Resultado do braço 0 — teste cego "antes" (2026-10-29)

Rodado uma única vez, depois do pré-registro (`793a8ed`): 6 modelos finais com k=3, N=4000 e os hiperparâmetros da linha de base (`06` → `08 --tag antes --n-components 4000`). Arquivos: `reports/v2/teste_cego_antes_*.json`, `tabelas_teste_cego_antes.tsv`. 294 seqs (147/147), 102 genes (46/56).

| Modelo | CV (genes misturados): ROC-AUC / MCC | Teste cego por gene: ROC-AUC [IC 95%] | MCC por gene | Sens. / Espec. por gene |
|---|---|---|---|---|
| SVM (principal) | 99,7 / 95,3 | 0,53 [0,42; 0,64] | 0,12 | 0,43 / 0,68 |
| LR | 99,6 / 94,7 | 0,55 [0,43; 0,66] | 0,01 | 0,46 / 0,55 |
| RF | 98,7 / 85,5 | 0,57 [0,45; 0,68] | 0,19 | 0,37 / 0,80 |
| GB | 98,5 / 87,5 | 0,58 [0,47; 0,69] | 0,15 | 0,41 / 0,73 |
| NB | 93,9 / 68,3 | 0,61 [0,50; 0,72] | 0,15 | 0,20 / 0,91 |
| kNN | 93,9 / 68,1 | 0,49 [0,42; 0,56] | −0,04 | 0,02 / 0,96 |

IC 95%: bootstrap por gene (5000 reamostragens, estratificado por classe). **Nenhum modelo se distingue do acaso** (todos os IC contêm 0,5; o NB no limite).

- **A CV de genes misturados não prevê a generalização:** ~99,7 de ROC-AUC na CV contra ~0,55 em genes novos. Confirma §4.1 — a CV mede reconhecimento de homólogos do treino.
- **Sem gradiente por nível de evidência** (SVM: A 3/11, B 1/4, C 7/15, D 9/16) → o fracasso não vem de rótulos fracos do nível D.
- **Negativos difíceis não são mais confundidos que os fáceis** (SVM, P(positivo) média: difícil 0,37, fácil 0,42, intermediário 0,45; positivos 0,41–0,48) → o modelo não aprendeu "imune × housekeeping"; para genes nunca vistos, devolve probabilidades pouco informativas, em torno de 0,4–0,5, qualquer que seja a classe.
- Coerente com a validação externa das 5 proteínas (INS, PTPN22 e RUNX3 erradas).
- **Leitura para o TCC:** resultado negativo, central e defensável — pré-registrado, teste separado por gene por construção. Os braços A e B seguem como registrados: o B testa diretamente se escolher hiperparâmetros com CV separada por gene muda isso.

### Resultado do braço B — otimizado com CV separada por gene (2026-10-30)

Otimização na VM de 2026-10-29 22:34 a 2026-10-30 20:54 UTC (~22 h; boosting levou ~17 h — tentativas de 6–8 min, bem acima do piloto, e uma ampliação de faixa: `gb_learning_rate` até 0,0005, +50 tentativas). Saídas em `reports/v2/otimizacao/por_gene/` e `models/v2/otimizacao/por_gene/`; teste cego rodado uma vez em 2026-10-30, depois da decisão de seguir com o braço A (para o resultado do B não influenciar essa decisão).

| Modelo | CV por gene: ROC-AUC / MCC | Teste cego por gene: ROC-AUC [IC 95%] | MCC | Limiar |
|---|---|---|---|---|
| **MLP (principal)** | 0,62 / 0,19 | 0,49 [0,38; 0,61] | −0,04 | 0,35 |
| Comitê | 0,61 / 0,17 | 0,53 [0,41; 0,64] | 0,07 | 0,50 |
| LR | 0,63 / 0,17 | 0,56 [0,44; 0,67] | 0,10 | 0,50 |
| Boosting | 0,60 / 0,16 | 0,59 [0,48; 0,70] | 0,15 | 0,49 |
| SVM | 0,58 / 0,15 | 0,55 [0,43; 0,66] | 0,00 | 0,48 |
| Árvores | 0,53 / 0,05 | 0,58 [0,46; 0,69] | 0,11 | 0,57 |

- **Teste cego continua no acaso** (todos os IC contêm 0,5); nenhuma melhora sobre o braço 0 (SVM 0,53, LR 0,55, RF 0,57, GB 0,58).
- **Achado metodológico central:** a CV **separada por gene** prevê o teste cego (CV ~0,53–0,63 × cego ~0,49–0,59); a CV de **genes misturados** não (CV ~0,99 × cego ~0,55). A CV agrupada é a estimativa honesta para genes novos.
- **A busca estabilizou:** o melhor da 1ª metade das tentativas já estava a ≤ 0,02 de MCC do melhor final (incerteza da média ±0,03), e seis famílias de modelos muito diferentes convergem para MCC ~0,15–0,19 / ROC-AUC ~0,60 → o limite é a informação da representação (composição de k-mers), não os hiperparâmetros. Trabalhos futuros: embeddings de modelos de linguagem de proteínas (ESM-2), contexto biológico além da sequência (interações, vias, expressão, GO — "culpa por associação"), mais genes positivos, rótulos mais limpos (nível D).

### Arquivos perdidos em 2026-10-10

Por volta do `git pull --rebase` de 2026-10-10 15:29, sumiram todos os não versionados: `CLAUDE.md`, `documentacao.md`, `src/00_fetch_uniprot.py`, `data/gene_lists/*`, `data/raw_v2/*` e alterações locais de `README.md`/`requirements.txt`. Causa não identificada. Recriados em 2026-10-11: este arquivo, as listas, os scripts 00 e os FASTAs v2. `documentacao.md` **não** foi recuperado.

---

## 7. TCC II — outras direções previstas

Registradas em "Trabalhos Futuros" (§6.1 do documento):

1. **Variar k** — k=2 (400 dims) e k=4 (160.000 dims) contra k=3.
2. **Refinar modelos** — grids mais amplos; redes neurais.
3. **Aplicar ao proteoma humano completo** (~20.000 proteínas) para triagem de candidatos.

Corrigir o vazamento por homólogos (§4.1) deve preceder qualquer conclusão nova.
