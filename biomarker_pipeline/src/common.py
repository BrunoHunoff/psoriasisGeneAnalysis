"""
Definições compartilhadas pela pipeline: datasets (TCC I / TCC II) e vetorização k-mer.

Cada dataset tem seus próprios diretórios de entrada E de saída, para que rodar o TCC II
nunca sobrescreva os artefatos e relatórios do TCC I (e vice-versa):

  tcc1 — dataset congelado do TCC I (data/raw + data/raw_negativos), saídas em
         data/processed, models/ e reports/ — os mesmos caminhos usados nos quadros do TCC I
  tcc2 — expansão Dand et al. (2025), saídas em data/processed_v2, models/v2 e reports/v2;
         tem conjunto de teste cego (data/teste_cego_v2 + data/teste_cego_negativos_v2)

Os scripts recebem --dataset {tcc1,tcc2} (padrão: DEFAULT_DATASET).

Representação k-mer (--k / --pattern): cada padrão é uma máscara sobre uma janela da
sequência — "111" é o k-mer contínuo de k=3, "1101" lê as posições 1, 2 e 4 (k=3 pulando
uma posição). Vários padrões de mesmo k e mesmo comprimento são combinados lado a lado.
k=3 contínuo (o do TCC I) usa os diretórios do dataset; as demais representações gravam
num subdiretório com o nome delas (ex.: data/processed_v2/k4/, reports/v2/k3_1101/).

Vetor final (--n-components): N > 0 → projeção gaussiana para N dimensões; 0 → sem
projeção (o vetor é a frequência de cada k-mer). Os arquivos levam a dimensão final no
nome (X_n{dim}.npy, results_{modelo}_n{dim}.json), então "sem projeção" com k=3 é n8000.
"""

import os
import re
import functools
import itertools

import numpy as np
from scipy import sparse

DATASETS = {
    "tcc1": {
        "pos_dir":       "data/raw",
        "neg_dir":       "data/raw_negativos",
        "proc_dir":      "data/processed",
        "models_dir":    "models",
        "reports_dir":   "reports",
        "blind_pos_dir": None,
        "blind_neg_dir": None,
    },
    "tcc2": {
        "pos_dir":       "data/raw_v2",
        "neg_dir":       "data/raw_negativos_v2",
        "proc_dir":      "data/processed_v2",
        "models_dir":    os.path.join("models", "v2"),
        "reports_dir":   os.path.join("reports", "v2"),
        "blind_pos_dir": "data/teste_cego_v2",
        "blind_neg_dir": "data/teste_cego_negativos_v2",
    },
}
DEFAULT_DATASET = "tcc2"

# Proteínas de validação externa do TCC I (data/input/), repetidas no TCC II para comparação
INPUT_DIR = "data/input"

K            = 3
AMINO        = "ACDEFGHIKLMNPQRSTVWY"
FASTA_EXTS   = (".fasta", ".fa")
RANDOM_STATE = 42

DEFAULT_PATTERNS = ("1" * K,)       # k=3 contínuo — a representação do TCC I
REP_NAME_RE      = re.compile(r"^k\d+(_[01]+(\+[01]+)*)?$")

# Teto de memória por bloco da matriz de projeção (float32). k=3 cabe num bloco só;
# k=4 com N=10.000 (160 mil × 10 mil, 6,4 GB) é gerado e aplicado em fatias.
PROJECTION_BLOCK_ELEMS = 128 * 1024 * 1024   # ~512 MB


def add_dataset_arg(parser) -> None:
    parser.add_argument(
        "--dataset", choices=list(DATASETS), default=DEFAULT_DATASET,
        help=f"Dataset: tcc1 (congelado do TCC I) ou tcc2 (Dand 2025). Padrão: {DEFAULT_DATASET}.",
    )


def add_representation_args(parser, n_components: bool = True, require_n_components: bool = False) -> None:
    parser.add_argument(
        "--k", type=int, default=K,
        help=f"Tamanho do k-mer contínuo (padrão: {K}). Ignorado se --pattern for dado.",
    )
    parser.add_argument(
        "--pattern", nargs="+", default=None, metavar="MASCARA",
        help="k-mer espaçado: máscara(s) de 0/1, ex. 1101 ou 1101 1011 (combinadas).",
    )
    if n_components:
        parser.add_argument(
            "--n-components", type=int, required=require_n_components,
            default=None if require_n_components else 1000,
            help="Dimensão da projeção gaussiana; 0 = sem projeção"
                 + (" (obrigatório)." if require_n_components else " (padrão: 1000)."),
        )


def patterns_from_args(args) -> tuple[str, ...]:
    """--k / --pattern → tupla de máscaras validadas."""
    patterns = tuple(args.pattern) if args.pattern else ("1" * args.k,)
    for p in patterns:
        if not re.fullmatch(r"1[01]*1|1", p):
            raise SystemExit(f"Máscara inválida: {p!r} (só 0/1, começando e terminando em 1).")
    if len({p.count("1") for p in patterns}) > 1 or len({len(p) for p in patterns}) > 1:
        raise SystemExit(f"Máscaras combinadas precisam ter o mesmo k e o mesmo comprimento: {patterns}")
    if len(set(patterns)) < len(patterns):
        raise SystemExit(f"Máscara repetida: {patterns}")
    return patterns


def rep_name(patterns: tuple[str, ...]) -> str:
    """Nome da representação: k3, k4, k3_1101, k3_1101+1011."""
    k = patterns[0].count("1")
    if len(patterns) == 1 and patterns[0] == "1" * k:
        return f"k{k}"
    return f"k{k}_" + "+".join(patterns)


def n_kmer_features(patterns: tuple[str, ...]) -> int:
    return len(patterns) * len(AMINO) ** patterns[0].count("1")


def vector_dim(patterns: tuple[str, ...], n_components: int) -> int:
    """Dimensão do vetor final: N, ou o número de k-mers se não houver projeção (N=0)."""
    return n_components if n_components else n_kmer_features(patterns)


def dataset_paths(name: str, patterns: tuple[str, ...] = DEFAULT_PATTERNS) -> dict:
    """Diretórios do dataset; representações além do k=3 contínuo ficam num subdiretório."""
    paths = dict(DATASETS[name])
    if patterns != DEFAULT_PATTERNS:
        sub = rep_name(patterns)
        for key in ("proc_dir", "models_dir", "reports_dir"):
            paths[key] = os.path.join(paths[key], sub)
    return paths


def x_path(proc_dir: str, dim: int) -> str:
    return os.path.join(proc_dir, f"X_n{dim}.npy")


def results_path(reports_dir: str, model: str, dim: int) -> str:
    return os.path.join(reports_dir, f"results_{model}_n{dim}.json")


def classifier_path(models_dir: str, model: str, n_components: int) -> str:
    return os.path.join(models_dir, f"classifier_{model}_n{n_components}.pkl")


def build_vocab(amino: str = AMINO, k: int = K) -> tuple[list[str], dict[str, int]]:
    """Gera todas as combinações de k aminoácidos. Retorna (lista, mapa kmer→índice)."""
    vocab = ["".join(c) for c in itertools.product(amino, repeat=k)]
    return vocab, {kmer: idx for idx, kmer in enumerate(vocab)}


@functools.lru_cache(maxsize=None)
def _vocab_index(k: int) -> dict[str, int]:
    return build_vocab(AMINO, k)[1]


def feature_names(patterns: tuple[str, ...]) -> list[str]:
    """Rótulo de cada coluna da matriz k-mer; posições puladas aparecem como "_" (ex.: AC_D)."""
    vocab, _ = build_vocab(AMINO, patterns[0].count("1"))
    names = []
    for p in patterns:
        for kmer in vocab:
            letters = iter(kmer)
            names.append("".join(next(letters) if c == "1" else "_" for c in p))
    return names


def sequence_to_kmer_matrix(seq: str, patterns: tuple[str, ...] = DEFAULT_PATTERNS) -> sparse.csr_matrix:
    """
    Constrói a matriz posicional one-hot (L-s+1, n_kmer_features) de uma sequência, em formato
    esparso (CSR), onde s é o comprimento da máscara. Cada padrão ocupa um bloco de 20^k colunas;
    em cada linha, cada padrão marca 1 na coluna do k-mer que ele lê naquela posição (com padrões
    combinados a linha tem um 1 por padrão). K-mers com caracteres fora do vocabulário resultam
    em zero naquele bloco.

    Com o padrão "111" é a mesma matriz (L-2, 8000) do TCC I, só armazenada sem os zeros: no
    TCC II a matriz global densa teria ~325 mil janelas × 8.000 colunas (~10 GB em float32).
    """
    k, span = patterns[0].count("1"), len(patterns[0])
    vocab_index = _vocab_index(k)
    n_windows = len(seq) - span + 1
    if n_windows <= 0:
        raise ValueError(f"Sequência curta demais para a máscara {patterns[0]}: len={len(seq)}")
    rows, cols = [], []
    for block, p in enumerate(patterns):
        offsets = [j for j, c in enumerate(p) if c == "1"]
        base = block * len(vocab_index)
        for i in range(n_windows):
            idx = vocab_index.get("".join(seq[i + j] for j in offsets))
            if idx is not None:
                rows.append(i)
                cols.append(base + idx)
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n_windows, n_kmer_features(patterns)),
                             dtype=np.float32)


def _projection_blocks(n_components: int, n_features: int) -> list[tuple[int, int]]:
    """Fatias de componentes da projeção; nenhuma com largura 1 (a média do numpy muda de ordem)."""
    width = max(2, PROJECTION_BLOCK_ELEMS // n_features)
    blocks = [(a, min(a + width, n_components)) for a in range(0, n_components, width)]
    if len(blocks) > 1 and blocks[-1][1] - blocks[-1][0] == 1:
        blocks[-2:] = [(blocks[-2][0], n_components)]
    return blocks


def kmer_vectors(matrices: list[sparse.csr_matrix], n_components: int,
                 random_state: int = RANDOM_STATE) -> np.ndarray:
    """
    Matrizes k-mer → vetores finais (n_seq, dim), uma linha por sequência.

    n_components > 0: projeção gaussiana de cada janela (8000 → N) seguida de mean pooling.
    A matriz de projeção é a mesma do GaussianRandomProjection(n_components, random_state=42)
    do scikit-learn — N(0, 1/N), gerada linha a linha pelo mesmo RandomState e convertida para
    float32 — e o resultado é idêntico bit a bit ao da versão anterior, que ajustava o
    projetor sobre a matriz global empilhada. Como a projeção não depende dos dados, projetar
    sequência por sequência dá o mesmo resultado; gerar em blocos evita guardar os 6,4 GB da
    projeção de k=4 com N=10.000. Sequências novas (07/08) são vetorizadas por esta mesma
    função, então treino e predição usam exatamente a mesma projeção.

    n_components == 0: sem projeção — mean pooling direto das linhas one-hot, isto é, a
    frequência relativa de cada k-mer na sequência.
    """
    n_features = matrices[0].shape[1]
    if not n_components:
        return np.vstack([np.asarray(m.sum(axis=0), dtype=np.float64).ravel() / m.shape[0]
                          for m in matrices]).astype(np.float32)
    if n_components >= n_features:
        raise ValueError(f"N={n_components} ≥ {n_features} k-mers: use n_components=0 (sem projeção).")
    rng = np.random.RandomState(random_state)
    scale = 1.0 / np.sqrt(n_components)
    X = np.empty((len(matrices), n_components), dtype=np.float32)
    for a, b in _projection_blocks(n_components, n_features):
        table = np.ascontiguousarray(
            rng.normal(loc=0.0, scale=scale, size=(b - a, n_features)).astype(np.float32).T
        )
        for i, m in enumerate(matrices):
            X[i, a:b] = np.asarray(m @ table).mean(axis=0)
    return X


def sequences_to_vectors(seqs: list[str], patterns: tuple[str, ...],
                         n_components: int) -> tuple[np.ndarray | None, list[int]]:
    """
    Vetoriza sequências novas exatamente como o treino (01 + 02). Retorna (X, índices
    vetorizados); sequências mais curtas que a máscara ficam de fora.
    """
    matrices, kept = [], []
    for i, seq in enumerate(seqs):
        try:
            matrices.append(sequence_to_kmer_matrix(seq, patterns))
        except ValueError:
            continue
        kept.append(i)
    if not matrices:
        return None, kept
    return kmer_vectors(matrices, n_components), kept
