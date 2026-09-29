"""
Módulo 00 — Coleta de proteínas negativas "difíceis" para data/raw_negativos/.

Negativo difícil = proteína que se parece com os biomarcadores da psoríase
(mesma família, mesma via imune ou biomarcador de outra doença imunomediada),
mas SEM evidência de associação com psoríase.

Fontes de candidatos:
  A. Parálogos humanos de cada biomarcador de data/raw/: membros da mesma família
     proteica anotada no UniProt (ou do mesmo domínio Pfam, quando não há família),
     ordenados por identidade de sequência com o biomarcador (alinhamento BLOSUM62)
  B. Genes com forte associação genética a outras doenças imunomediadas
     (Open Targets Platform, datatype genetic_association)
  C. Citocinas e receptores de citocinas humanos (UniProt Swiss-Prot:
     keyword KW-0202 e GO:0004896)

Critério de exclusão (qualquer um exclui o gene):
  1. Gene presente nos positivos (data/raw/) ou nas entradas de teste (data/input/)
  2. Open Targets — psoríase ou artrite psoriásica com score global >= OT_PSO_MAX_SCORE,
     ou qualquer evidência genética/clínica/modelo animal, ou expressão diferencial
     na pele psoriásica com rna_expression >= OT_PSO_MAX_RNA
  3. GWAS Catalog — gene mapeado em qualquer associação com psoríase (e subtipos) ou
     artrite psoriásica, inclusive estudos multi-doença que incluem psoríase
  4. UniProt — entrada humana revisada cujo texto cita psoriasis/psoriatic/psoriasin

Sequências: somente UniProtKB/Swiss-Prot (revisadas), proteína canônica humana
+ ortólogos de mamíferos com o mesmo gene primário, até SEQS_PER_GENE por gene
(mesmo padrão dos arquivos já existentes).

Saídas:
  data/raw_negativos/{GENE}_{categoria}.fasta
  reports/negativos/negativos_manifest.tsv — gene, categoria, fonte e evidência de cada arquivo
  reports/negativos/psoriase_exclusao.tsv  — genes excluídos por associação com psoríase e motivo
"""

import os
import re
import sys
import csv
import json
import time
import logging
import urllib.error
import urllib.parse
import urllib.request

from Bio.Align import PairwiseAligner, substitution_matrices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

RAW_POS_DIR    = "data/raw"
INPUT_DIR      = "data/input"
RAW_NEG_DIR    = "data/raw_negativos"
REPORT_DIR     = os.path.join("reports", "negativos")
MANIFEST_PATH  = os.path.join(REPORT_DIR, "negativos_manifest.tsv")
EXCLUSION_PATH = os.path.join(REPORT_DIR, "psoriase_exclusao.tsv")
FASTA_EXTS     = (".fasta", ".fa")

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"
OT_GRAPHQL     = "https://api.platform.opentargets.org/api/v4/graphql"
# Endpoint de download usado pelo próprio site do GWAS Catalog (inclui traços-filho)
GWAS_DOWNLOAD  = ("https://www.ebi.ac.uk/gwas/api/v2/efotraits/{trait}/associations/download"
                  "?includeBgTraits=false&includeChildTraits=true")

HTTP_TIMEOUT   = 300
HTTP_RETRIES   = 4
HTTP_BACKOFF_S = 5

# --- Exclusão (psoríase) ---
PSORIASIS_OT_IDS   = ["MONDO_0005083", "MONDO_0011849"]  # psoríase, artrite psoriásica
OT_PSO_MAX_SCORE   = 0.05     # acima disso já é associação relevante mesmo só com literatura
OT_PSO_MAX_RNA     = 0.1      # ~25% mais diferencialmente expressos na lesão (S100A7, DEFB4A, IL36A...)
OT_WEAK_DATATYPES  = {"literature"}
OT_RNA_DATATYPE    = "rna_expression"
OT_PAGE_SIZE       = 1000
GWAS_PSO_TRAITS    = ["MONDO_0005083", "MONDO_0011849"]  # + filhos: vulgar, pustular, ...
GWAS_GENE_COLUMN   = "mappedGenes"
GWAS_NO_GENE       = "-"
UNIPROT_PSO_QUERY  = ("(psoriasis OR psoriatic OR psoriasin) "
                      "AND (organism_id:9606) AND (reviewed:true)")

# --- Fonte A: parálogos (família UniProt / domínio Pfam) ---
PARALOG_TOP_N        = 5
PARALOG_MIN_IDENTITY = 20.0   # % de resíduos idênticos sobre o tamanho da proteína menor
FAMILY_SEPARATOR     = ", "   # "Protein kinase superfamily, Tyr protein kinase family, JAK subfamily"
SUPERFAMILY_WORD     = "superfamily"
PFAM_MAX_MEMBERS     = 60     # ignora domínios genéricos (Ig, F-box, RING...) com muitos membros
ALIGN_MATRIX         = "BLOSUM62"
ALIGN_GAP_OPEN       = -10.0
ALIGN_GAP_EXTEND     = -0.5
ALIGN_ALPHABET       = set("ACDEFGHIKLMNPQRSTVWY")
ALIGN_UNKNOWN        = "X"

# --- Fonte B: outras doenças imunomediadas (Open Targets) ---
OTHER_DISEASES = {
    "MONDO_0005011": "Crohn",
    "MONDO_0005101": "Colite_Ulcerativa",
    "MONDO_0008383": "Artrite_Reumatoide",
    "MONDO_0007915": "Lupus",
    "MONDO_0005147": "Diabetes_Tipo1",
    "MONDO_0005130": "Doenca_Celiaca",
    "MONDO_0004980": "Dermatite_Atopica",
    "MONDO_0004979": "Asma",
    "MONDO_0008661": "Vitiligo",
    "MONDO_0005340": "Alopecia_Areata",
    "MONDO_0005364": "Doenca_de_Graves",
    "MONDO_0005100": "Esclerose_Sistemica",
    "MONDO_0010030": "Sjogren",
    "MONDO_0005388": "Colangite_Biliar_Primaria",
    "MONDO_0005306": "Espondilite_Anquilosante",
}
DISEASE_MIN_GENETIC = 0.5
DISEASE_TOP_N       = 20

# --- Fonte C: citocinas / receptores (UniProt) ---
IMMUNE_QUERIES = {
    "KW-0202 Cytokine":            "(keyword:KW-0202) AND (organism_id:9606) AND (reviewed:true)",
    "GO:0004896 cytokine receptor": "(go:0004896) AND (organism_id:9606) AND (reviewed:true)",
}

# --- Sequências ---
SEQS_PER_GENE     = 4
MIN_LENGTH        = 50      # mesmo intervalo de config.yaml (data.min_length/max_length)
MAX_LENGTH        = 2000
HUMAN_TAXID       = 9606
MAMMALIA_TAXID    = 40674
SPECIES_PRIORITY  = [9606, 10090, 10116, 9913, 9823, 9615, 9544, 9598,
                     9796, 9986, 9685, 9940]
UNIPROT_GENE_BATCH = 40
UNIPROT_ACC_BATCH  = 100
UNIPROT_PAGE_SIZE  = 500
HUMAN_REVIEWED     = f"(organism_id:{HUMAN_TAXID}) AND (reviewed:true)"

CAT_PARALOG  = "paralogo_{pos}_dificil"
CAT_IMMUNE   = "nao_ligados_psioriase_dificil"   # mesmo sufixo dos arquivos existentes
GN_PATTERN   = re.compile(r"\bGN=(\S+)")
OX_PATTERN   = re.compile(r"\bOX=(\d+)")
NEXT_LINK    = re.compile(r'<([^>]+)>;\s*rel="next"')


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def http(url: str, data: bytes | None = None, headers: dict | None = None) -> tuple[str, dict]:
    """GET/POST com retentativas. Retorna (corpo, cabeçalhos)."""
    last_err = None
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers or {})
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                return resp.read().decode(), dict(resp.headers)
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            log.warning("Falha HTTP (%d/%d) %s: %s", attempt, HTTP_RETRIES, url[:120], e)
            time.sleep(HTTP_BACKOFF_S * attempt)
    raise RuntimeError(f"Falha definitiva em {url}: {last_err}")


def http_json(url: str, payload: dict | None = None) -> dict:
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
    body, _ = http(url, data, headers)
    return json.loads(body)


def uniprot_tsv(query: str, fields: str) -> list[dict]:
    """Busca paginada no UniProt, retorna linhas TSV como dicts."""
    params = urllib.parse.urlencode({
        "query": query, "format": "tsv", "fields": fields, "size": UNIPROT_PAGE_SIZE,
    })
    url = f"{UNIPROT_SEARCH}?{params}"
    rows = []
    while url:
        body, headers = http(url, headers={"Accept": "text/plain"})
        lines = body.splitlines()
        if lines:
            reader = csv.DictReader(lines, delimiter="\t")
            rows.extend(reader)
        match = NEXT_LINK.search(headers.get("Link", "") or headers.get("link", ""))
        url = match.group(1) if match else None
    return rows


def ot_query(query: str, variables: dict) -> dict:
    res = http_json(OT_GRAPHQL, {"query": query, "variables": variables})
    if res.get("errors"):
        raise RuntimeError(f"Open Targets: {res['errors']}")
    return res["data"]


OT_ASSOC_QUERY = """
query($id: String!, $index: Int!, $size: Int!) {
  disease(efoId: $id) {
    name
    associatedTargets(page: {index: $index, size: $size}) {
      count
      rows { score target { approvedSymbol } datatypeScores { id score } }
    }
  }
}"""


def ot_disease_targets(disease_id: str) -> tuple[str, list[dict]]:
    """Todas as associações alvo-doença (diretas + indiretas) de uma doença no Open Targets."""
    rows, index, name = [], 0, disease_id
    while True:
        data = ot_query(OT_ASSOC_QUERY, {"id": disease_id, "index": index, "size": OT_PAGE_SIZE})
        disease = data["disease"]
        if disease is None:
            log.error("Doença %s não encontrada no Open Targets.", disease_id)
            return name, []
        name = disease["name"]
        page = disease["associatedTargets"]
        rows.extend(page["rows"])
        if not page["rows"] or len(rows) >= page["count"]:
            break
        index += 1
    return name, [
        {
            "symbol": r["target"]["approvedSymbol"].upper(),
            "score": r["score"],
            "datatypes": {d["id"]: d["score"] for d in r["datatypeScores"]},
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Genes locais
# ---------------------------------------------------------------------------

def fasta_headers(directory: str) -> dict[str, list[str]]:
    """{nome_arquivo: [cabeçalhos]} de todos os FASTA do diretório."""
    out = {}
    if not os.path.isdir(directory):
        return out
    for fname in sorted(os.listdir(directory)):
        if os.path.splitext(fname)[1].lower() not in FASTA_EXTS:
            continue
        with open(os.path.join(directory, fname)) as f:
            out[fname] = [line.strip() for line in f if line.startswith(">")]
    return out


def genes_in_headers(headers: list[str]) -> set[str]:
    return {m.group(1).upper() for h in headers for m in GN_PATTERN.finditer(h)}


def positive_seed_genes(pos_files: dict[str, list[str]]) -> dict[str, str]:
    """
    {arquivo_stem: símbolo humano} para cada biomarcador em data/raw/.
    Usa o GN= da sequência humana cujo nome casa com o arquivo; senão, o primeiro GN humano.
    """
    seeds = {}
    for fname, headers in pos_files.items():
        stem = os.path.splitext(fname)[0].upper()
        human = [GN_PATTERN.search(h).group(1).upper() for h in headers
                 if OX_PATTERN.search(h) and OX_PATTERN.search(h).group(1) == str(HUMAN_TAXID)
                 and GN_PATTERN.search(h)]
        seeds[stem] = stem if stem in human else (human[0] if human else stem)
    return seeds


# ---------------------------------------------------------------------------
# Exclusão
# ---------------------------------------------------------------------------

def build_psoriasis_exclusion() -> tuple[dict[str, list[str]], dict[str, float]]:
    """
    Retorna ({GENE: [motivos]} dos genes com evidência relevante de associação com psoríase,
    {GENE: maior score Open Targets psoríase/artrite psoriásica} de todos os alvos listados).
    """
    reasons: dict[str, list[str]] = {}
    ot_scores: dict[str, float] = {}

    def add(gene: str, reason: str):
        reasons.setdefault(gene.upper(), []).append(reason)

    for did in PSORIASIS_OT_IDS:
        name, rows = ot_disease_targets(did)
        n_before = len(reasons)
        for r in rows:
            ot_scores[r["symbol"]] = max(ot_scores.get(r["symbol"], 0.0), r["score"])
            strong_types = sorted(
                k for k in set(r["datatypes"]) - OT_WEAK_DATATYPES
                if k != OT_RNA_DATATYPE or r["datatypes"][k] >= OT_PSO_MAX_RNA
            )
            if r["score"] >= OT_PSO_MAX_SCORE or strong_types:
                detail = ",".join(f"{k}={r['datatypes'][k]:.2f}" for k in strong_types)
                add(r["symbol"], f"OpenTargets:{name}:score={r['score']:.3f}"
                                 + (f"[{detail}]" if detail else ""))
        log.info("  Open Targets %s (%s): %d alvos associados, %d novos excluídos",
                 did, name, len(rows), len(reasons) - n_before)

    for trait in GWAS_PSO_TRAITS:
        body, _ = http(GWAS_DOWNLOAD.format(trait=trait), headers={"Accept": "*/*"})
        assoc = list(csv.DictReader(body.splitlines(), delimiter="	"))
        genes = {g.strip() for a in assoc for g in a[GWAS_GENE_COLUMN].split(",")
                 if g.strip() and g.strip() != GWAS_NO_GENE}
        for g in genes:
            add(g, f"GWASCatalog:{trait}")
        log.info("  GWAS Catalog %s (+filhos): %d associações, %d genes mapeados",
                 trait, len(assoc), len(genes))

    rows = uniprot_tsv(UNIPROT_PSO_QUERY, "accession,gene_primary")
    for r in rows:
        if r["Gene Names (primary)"]:
            add(r["Gene Names (primary)"], f"UniProt:{r['Entry']}:cita psoríase")
    log.info("  UniProt: %d entradas humanas citam psoríase", len(rows))

    return reasons, ot_scores


# ---------------------------------------------------------------------------
# Candidatos
# ---------------------------------------------------------------------------

def make_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load(ALIGN_MATRIX)
    aligner.open_gap_score = ALIGN_GAP_OPEN
    aligner.extend_gap_score = ALIGN_GAP_EXTEND
    aligner.end_gap_score = 0.0   # semi-global: pontas não penalizam proteínas de tamanhos diferentes
    return aligner


def percent_identity(aligner: PairwiseAligner, a: str, b: str) -> float:
    """% de resíduos idênticos no alinhamento, sobre o tamanho da proteína menor."""
    def clean(seq: str) -> str:
        return "".join(c if c in ALIGN_ALPHABET else ALIGN_UNKNOWN for c in seq)
    alignment = aligner.align(clean(a), clean(b))[0]
    return 100.0 * alignment.counts().identities / min(len(a), len(b))


def human_members(query: str) -> dict[str, str]:
    """{GENE: sequência} das entradas humanas revisadas que casam com a query."""
    rows = uniprot_tsv(f"({query}) AND {HUMAN_REVIEWED}", "accession,gene_primary,sequence")
    return {r["Gene Names (primary)"].upper(): r["Sequence"] for r in rows if r["Gene Names (primary)"]}


def family_levels(families: str) -> list[tuple[str, str]]:
    """
    Níveis da família UniProt, do mais específico ao mais amplo, como (rótulo, query).
    Cada query exige também os níveis acima (senão "Type 2 subfamily" casa com HDACs).
    Um nível "superfamily" só entra quando é a única anotação (ex.: IL-6 superfamily).
    """
    levels = [lv.strip() for lv in families.split(FAMILY_SEPARATOR) if lv.strip()]
    out = []
    for depth in range(len(levels), 0, -1):
        path = levels[:depth]
        if SUPERFAMILY_WORD in path[-1].lower() and len(levels) > 1:
            break
        out.append((" > ".join(path), " AND ".join(f'family:"{lv}"' for lv in path)))
    return out


def paralog_candidates(seeds: dict[str, str], blocked: set[str]) -> list[dict]:
    """
    Parálogos humanos de cada biomarcador, ordenados por identidade de sequência.

    Membros da família UniProt do biomarcador, subindo de nível enquanto houver menos
    de PARALOG_TOP_N genes utilizáveis (fora de `blocked`). Sem família anotada, usa
    os domínios Pfam do biomarcador (ignorando domínios com > PFAM_MAX_MEMBERS membros).
    """
    symbols = sorted(set(seeds.values()))
    gene_q = " OR ".join(f'gene_exact:"{g}"' for g in symbols)
    rows = uniprot_tsv(f"({gene_q}) AND {HUMAN_REVIEWED}",
                       "accession,gene_primary,protein_families,xref_pfam,sequence")
    info = {r["Gene Names (primary)"].upper(): r for r in rows if r["Gene Names (primary)"]}
    aligner = make_aligner()

    out = []
    for stem, symbol in seeds.items():
        seed = info.get(symbol)
        if seed is None:
            log.warning("  %s (%s): entrada humana não encontrada no UniProt", stem, symbol)
            continue

        members: dict[str, tuple[str, str]] = {}   # gene -> (sequência, origem)
        for label, query in family_levels(seed["Protein families"]):
            for gene, seq in human_members(query).items():
                members.setdefault(gene, (seq, f"familia UniProt '{label}'"))
            if sum(g not in blocked and g != symbol for g in members) >= PARALOG_TOP_N:
                break
        if not seed["Protein families"]:
            for pfam in filter(None, seed["Pfam"].split(";")):
                found = human_members(f"xref:pfam-{pfam}")
                if len(found) > PFAM_MAX_MEMBERS:
                    log.info("  %s: domínio %s ignorado (%d membros)", stem, pfam, len(found))
                    continue
                for gene, seq in found.items():
                    members.setdefault(gene, (seq, f"dominio Pfam {pfam}"))

        ranked = []
        for gene, (seq, origin) in members.items():
            if gene == symbol or gene in blocked or not MIN_LENGTH <= len(seq) <= MAX_LENGTH:
                continue
            pid = percent_identity(aligner, seed["Sequence"], seq)
            if pid >= PARALOG_MIN_IDENTITY:
                ranked.append((pid, gene, origin))
        ranked.sort(reverse=True)
        log.info("  %s (%s): %d membros da família/domínio, %d utilizáveis com identidade >= %.0f%%",
                 stem, symbol, len(members), len(ranked), PARALOG_MIN_IDENTITY)

        for pid, gene, origin in ranked:
            out.append({
                "gene": gene,
                "category": CAT_PARALOG.format(pos=stem),
                "group": stem,
                "cap": PARALOG_TOP_N,
                "source": "UniProt (familia/Pfam) + alinhamento BLOSUM62",
                "evidence": f"{origin}; identidade={pid:.1f}% com {symbol}",
            })
    return out


def disease_candidates() -> list[dict]:
    """Genes com forte associação genética a outras doenças imunomediadas (Open Targets)."""
    out = []
    for did, label in OTHER_DISEASES.items():
        name, rows = ot_disease_targets(did)
        strong = sorted(
            (r for r in rows if r["datatypes"].get("genetic_association", 0) >= DISEASE_MIN_GENETIC),
            key=lambda r: -r["datatypes"]["genetic_association"],
        )
        log.info("  %s (%s): %d alvos, %d com genetic_association >= %.2f",
                 did, name, len(rows), len(strong), DISEASE_MIN_GENETIC)
        for r in strong:
            out.append({
                "gene": r["symbol"],
                "category": label,
                "group": label,
                "cap": DISEASE_TOP_N,
                "source": f"Open Targets ({did} {name})",
                "evidence": (f"genetic_association={r['datatypes']['genetic_association']:.2f}; "
                             f"score_global={r['score']:.3f}"),
            })
    return out


def immune_candidates() -> list[dict]:
    """Citocinas e receptores de citocinas humanos revisados (UniProt)."""
    out = []
    for label, query in IMMUNE_QUERIES.items():
        rows = uniprot_tsv(query, "accession,gene_primary")
        log.info("  UniProt %s: %d entradas", label, len(rows))
        for r in sorted(rows, key=lambda r: r["Gene Names (primary)"]):
            if r["Gene Names (primary)"]:
                out.append({
                    "gene": r["Gene Names (primary)"].upper(),
                    "category": CAT_IMMUNE,
                    "group": CAT_IMMUNE,
                    "cap": None,
                    "source": f"UniProt ({label})",
                    "evidence": f"{r['Entry']}",
                })
    return out


# ---------------------------------------------------------------------------
# Sequências
# ---------------------------------------------------------------------------

def species_rank(taxid: int) -> int:
    return SPECIES_PRIORITY.index(taxid) if taxid in SPECIES_PRIORITY else len(SPECIES_PRIORITY)


def select_sequences(genes: list[str]) -> dict[str, list[dict]]:
    """
    Para cada gene: entrada humana Swiss-Prot + ortólogos de mamíferos revisados
    com o mesmo gene primário, tamanho em [MIN_LENGTH, MAX_LENGTH].
    Genes sem entrada humana válida ficam de fora.
    """
    selected = {}
    for i in range(0, len(genes), UNIPROT_GENE_BATCH):
        batch = genes[i:i + UNIPROT_GENE_BATCH]
        gene_q = " OR ".join(f'gene_exact:"{g}"' for g in batch)
        query = f"({gene_q}) AND (reviewed:true) AND (taxonomy_id:{MAMMALIA_TAXID})"
        rows = uniprot_tsv(query, "accession,gene_primary,organism_id,length")
        by_gene: dict[str, list[dict]] = {}
        for r in rows:
            gene = (r["Gene Names (primary)"] or "").upper()
            if gene not in batch:
                continue
            length = int(r["Length"])
            if not MIN_LENGTH <= length <= MAX_LENGTH:
                continue
            by_gene.setdefault(gene, []).append({
                "accession": r["Entry"], "taxid": int(r["Organism (ID)"]), "length": length,
            })
        for gene in batch:
            entries = by_gene.get(gene, [])
            if not any(e["taxid"] == HUMAN_TAXID for e in entries):
                continue
            entries.sort(key=lambda e: (species_rank(e["taxid"]), e["accession"]))
            chosen, seen_tax = [], set()
            for e in entries:
                if e["taxid"] in seen_tax:
                    continue
                chosen.append(e)
                seen_tax.add(e["taxid"])
                if len(chosen) == SEQS_PER_GENE:
                    break
            selected[gene] = chosen
        log.info("  UniProt: %d/%d genes processados", min(i + UNIPROT_GENE_BATCH, len(genes)), len(genes))
    return selected


def fetch_fasta(accessions: list[str]) -> dict[str, str]:
    """{accession: registro FASTA completo (cabeçalho UniProt + sequência em linhas de 60)}."""
    out = {}
    for i in range(0, len(accessions), UNIPROT_ACC_BATCH):
        batch = accessions[i:i + UNIPROT_ACC_BATCH]
        query = " OR ".join(f"accession:{a}" for a in batch)
        params = urllib.parse.urlencode({"query": query, "format": "fasta"})
        body, _ = http(f"{UNIPROT_STREAM}?{params}", headers={"Accept": "text/plain"})
        for rec in body.split(">")[1:]:
            acc = rec.split("|")[1]
            out[acc] = ">" + rec.rstrip("\n") + "\n"
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.isdir(RAW_POS_DIR) or not os.path.isdir(RAW_NEG_DIR):
        log.error("Rode a partir de biomarker_pipeline/ (precisa de %s e %s).", RAW_POS_DIR, RAW_NEG_DIR)
        sys.exit(1)

    # Arquivos gerados numa execução anterior (listados no manifesto) são recriados;
    # os demais arquivos de raw_negativos/ nunca são alterados.
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            previous = [row["arquivo"] for row in csv.DictReader(f, delimiter="\t")]
        for fname in previous:
            path = os.path.join(RAW_NEG_DIR, fname)
            if os.path.exists(path):
                os.remove(path)
        log.info("Removidos %d arquivos da execução anterior (manifesto).", len(previous))

    pos_files = fasta_headers(RAW_POS_DIR)
    input_files = fasta_headers(INPUT_DIR)
    neg_files = fasta_headers(RAW_NEG_DIR)
    if not pos_files:
        log.error("Nenhum FASTA positivo em %s.", RAW_POS_DIR)
        sys.exit(1)

    seeds = positive_seed_genes(pos_files)
    positive_genes = set(seeds) | set(seeds.values())
    for headers in list(pos_files.values()) + list(input_files.values()):
        positive_genes |= genes_in_headers(headers)
    existing_neg = {os.path.splitext(f)[0].split("_")[0].upper() for f in neg_files}
    log.info("Positivos/entradas: %d genes; negativos já existentes: %d", len(positive_genes), len(existing_neg))

    log.info("Montando lista de exclusão (psoríase)...")
    exclusion, ot_pso_scores = build_psoriasis_exclusion()
    for g in positive_genes:
        exclusion.setdefault(g, []).append("positivo/entrada do projeto")
    log.info("Genes excluídos por associação com psoríase: %d", len(exclusion))

    flagged = sorted(g for g in existing_neg if g in exclusion)
    for g in flagged:
        log.warning("Negativo EXISTENTE com evidência de psoríase: %s -> %s", g, "; ".join(exclusion[g]))

    log.info("Fonte A — parálogos dos biomarcadores (família UniProt / Pfam)...")
    cand_a = paralog_candidates(seeds, set(exclusion) | existing_neg)
    log.info("Fonte B — outras doenças imunomediadas (Open Targets)...")
    cand_b = disease_candidates()
    log.info("Fonte C — citocinas e receptores (UniProt)...")
    cand_c = immune_candidates()

    # Filtro de exclusão + deduplicação (ordem de prioridade: A, B, C)
    filtered, seen = [], set()
    for c in cand_a + cand_b + cand_c:
        g = c["gene"]
        if g in seen or g in exclusion or g in existing_neg:
            continue
        seen.add(g)
        filtered.append(c)
    log.info("Candidatos após exclusão/deduplicação: %d", len(filtered))

    log.info("Selecionando sequências no UniProt (Swiss-Prot, mamíferos)...")
    sequences = select_sequences([c["gene"] for c in filtered])

    # Limite por grupo (A: por biomarcador, B: por doença) aplicado só sobre genes com sequência
    count_per_group: dict[str, int] = {}
    chosen = []
    for c in filtered:
        if c["gene"] not in sequences:
            continue
        n = count_per_group.get(c["group"], 0)
        if c["cap"] is not None and n >= c["cap"]:
            continue
        count_per_group[c["group"]] = n + 1
        chosen.append(c)
    log.info("Genes selecionados: %d", len(chosen))

    accessions = [e["accession"] for c in chosen for e in sequences[c["gene"]]]
    fasta = fetch_fasta(accessions)

    manifest = []
    for c in chosen:
        entries = [e for e in sequences[c["gene"]] if e["accession"] in fasta]
        if not entries:
            continue
        fname = f"{c['gene']}_{c['category']}.fasta"
        with open(os.path.join(RAW_NEG_DIR, fname), "w", newline="\n") as f:
            for e in entries:
                f.write(fasta[e["accession"]])
        manifest.append({
            "arquivo": fname,
            "gene": c["gene"],
            "categoria": c["category"],
            "fonte": c["source"],
            "evidencia": c["evidence"],
            "n_seqs": len(entries),
            "acessos_uniprot": ",".join(e["accession"] for e in entries),
            "tamanho_humano": next(e["length"] for e in entries if e["taxid"] == HUMAN_TAXID),
            "ot_score_psoriase": f"{ot_pso_scores.get(c['gene'], 0.0):.3f}",
        })

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest)
    with open(EXCLUSION_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["gene", "negativo_existente", "motivos"])
        for g in sorted(exclusion):
            writer.writerow([g, "sim" if g in existing_neg else "", " | ".join(exclusion[g])])

    by_cat: dict[str, int] = {}
    for m in manifest:
        key = "paralogo" if m["categoria"].startswith("paralogo_") else m["categoria"]
        by_cat[key] = by_cat.get(key, 0) + 1
    log.info("Arquivos gravados em %s: %d (%d sequências)", RAW_NEG_DIR, len(manifest),
             sum(m["n_seqs"] for m in manifest))
    for k, v in sorted(by_cat.items(), key=lambda x: -x[1]):
        log.info("  %-35s %d", k, v)
    log.info("Manifesto: %s | Exclusões: %s", MANIFEST_PATH, EXCLUSION_PATH)


if __name__ == "__main__":
    main()
