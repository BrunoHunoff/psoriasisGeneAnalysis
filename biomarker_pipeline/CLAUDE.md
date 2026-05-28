# CLAUDE.md — Guia para o assistente IA neste projeto

## O que é este projeto

Pipeline para construção do **perfil vetorial** de biomarcadores proteicos da
psoríase (TCC de Engenharia de Software). Abordagem: perfil posicional, não classificação.

Fluxo: FASTA → matriz posicional one-hot k-mer → SVD por proteína → mean pooling → vetor fixo → perfil médio.

## Estrutura de módulos (Etapa 1)

| Módulo | Entrada | Saída |
|--------|---------|-------|
| `01_kmer_matrix.py` | `data/raw/*.fasta` | `data/processed/matrices/{stem}_matrix.npy` (L-2, 8000), `kmer_vocab.json`, `sequence_ids.json` |
| `02_svd_per_protein.py` | `matrices/*.npy` | `data/processed/vectors/{stem}_vector.npy` (N,), `models/svd_models/{stem}_svd.pkl`, `svd_summary.json`, `scree_plot_*.png` |
| `03_profile.py` | `vectors/*.npy`, `sequence_ids.json` | `profile_mean.npy`, `profile_std.npy`, `profile_summary.json`, `biomarker_space.png` |

## Abordagem científica (confirmada pelo orientador)

Cada proteína gera uma **matriz posicional one-hot** (L-2, 8000):
- Cada linha i = janela na posição i → 1 na coluna do 3-mer encontrado, 0 nas demais
- O SVD comprime essa matriz em (L-2, N) → a média das linhas (mean pooling) colapsa para (N,)
- Isso é a "projeção 3D→2D": rich positional object → fixed comparable vector

## Biomarcadores da psoríase (data/raw/)

7 proteínas humanas (UniProt), arquivos com sufixo `_PROTEINA.fasta` ou similar:
TRAF3IP2, TNIP1, TNFAIP3, IL23R, IL23A, IL12B, HLA-C

## Limitação conhecida do N

A menor proteína (IL23A, L=189) limita N global a máx 186.
Com matrizes one-hot esparsas, 90% de variância não é atingível com N fixo.
O código usa o maior N válido globalmente e reporta a variância real por proteína.

## Convenções obrigatórias

- **Sem valores hardcoded** — constantes nomeadas no topo de cada script
- **`logging`** em vez de `print` em todos os scripts `.py`
- **`random_state=42`** onde aplicável
- Cada módulo auto-suficiente: lê do disco, escreve no disco
- `data/raw/` vazia → `sys.exit(1)` com mensagem clara, sem dados sintéticos
- Não há split treino/teste — não é classificação

## Como rodar

```bash
bash run_part1.sh
```

O script ativa o venv automaticamente (`venv/`). Python = python3.

## O que NÃO fazer

- Não usar `print` — use `log.info()`, `log.warning()`, `log.error()`
- Não hardcodar paths — use constantes no topo e `os.path.join()`
- Não passar dados em memória entre módulos
- Não gerar dados sintéticos quando `data/raw/` estiver vazia
- Não fazer classificação — essa etapa é de perfil/similaridade
