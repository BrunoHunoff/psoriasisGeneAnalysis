"""
Módulo 00b — Coleta de sequências positivas (e homólogos) no UniProt.

Para cada gene de uma lista (ex.: saída de 00_build_gene_lists.py), baixa a entrada
humana Swiss-Prot e ortólogos revisados do mesmo gene em outras espécies, com os
MESMOS filtros usados pela coleta de negativos (00_fetch_negatives.py), para que
taxonomia e tamanho não virem atalho entre as classes:

  - reviewed:true, fragment:false, dentro de --taxon (default Mammalia)
  - tamanho em [MIN_LENGTH, MAX_LENGTH]
  - sem resíduos ambíguos X/Z/B (critério da §3.3.1 da documentação)
  - entrada humana obrigatória; uma sequência por espécie; até --max-seqs por gene
  - prioridade de espécies SPECIES_PRIORITY; desempate por accession (determinístico)

Validação de símbolo: a busca gene_exact: do UniProt também casa SINÔNIMOS e traz
proteínas de outros genes (NOS2→NANOS2, PRDX5→Prdx6, FBXL19→FBXL19-AS1...). Só é aceita
entrada cujo nome PRIMÁRIO é o símbolo pedido ou uma renomeação declarada em GENE_ALIASES.
Descartes vão para o log e para o relatório — ao adicionar genes, confira: um renomeado
legítimo aparece ali e precisa entrar em GENE_ALIASES.

Saídas:
  {out-dir}/{GENE}.fasta                        — cabeçalho UniProt, sequência em linhas de 60
  reports/coleta/{nome do out-dir}.json         — status por gene, organismos e descartes

Uso:
    python src/00_fetch_uniprot.py --genes data/gene_lists/positivos_dand2025_estabelecidos_tierA.txt --out-dir data/raw_v2
    python src/00_fetch_uniprot.py --genes data/gene_lists/teste_cego_dand2025_ineditos.txt --out-dir data/teste_cego_v2
"""

import os
import re
import ssl
import sys
import csv
import json
import time
import logging
import argparse
import urllib.error
import urllib.parse
import urllib.request

import certifi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

FROZEN_DIR      = "data/raw"          # dataset do TCC I — nunca escrever aqui
REPORT_DIR      = os.path.join("reports", "coleta")

UNIPROT_SEARCH  = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_STREAM  = "https://rest.uniprot.org/uniprotkb/stream"
HTTP_TIMEOUT    = 300
HTTP_RETRIES    = 4
HTTP_BACKOFF_S  = 5
SSL_CONTEXT     = ssl.create_default_context(cafile=certifi.where())

# Mesmos valores de 00_fetch_negatives.py
MAX_SEQS_PER_GENE = 10              # mesmo teto de homólogos do TCC I
MIN_LENGTH        = 50
MAX_LENGTH        = 2000
HUMAN_TAXID       = 9606
MAMMALIA_TAXID    = 40674
SPECIES_PRIORITY  = [9606, 10090, 10116, 9913, 9823, 9615, 9544, 9598,
                     9796, 9986, 9685, 9940]
INVALID_RESIDUES  = set("XZB")
UNIPROT_GENE_BATCH = 40
UNIPROT_ACC_BATCH  = 100
UNIPROT_PAGE_SIZE  = 500

# Símbolo pedido → nome primário atual no UniProt (renomeações legítimas)
GENE_ALIASES = {
    "DDX58":   "RIGI",
    "PTRF":    "CAVIN1",
    "C1orf68": "KPLCE",
    "IL28RA":  "IFNLR1",
    "ADC":     "AZIN2",
}

NEXT_LINK = re.compile(r'<([^>]+)>;\s*rel="next"')


# ---------------------------------------------------------------------------
# HTTP / UniProt
# ---------------------------------------------------------------------------

def http(url: str, headers: dict | None = None) -> tuple[str, dict]:
    """GET com retentativas. Retorna (corpo, cabeçalhos)."""
    last_err = None
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT, context=SSL_CONTEXT) as resp:
                return resp.read().decode(), dict(resp.headers)
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            log.warning("Falha HTTP (%d/%d) %s: %s", attempt, HTTP_RETRIES, url[:120], e)
            time.sleep(HTTP_BACKOFF_S * attempt)
    raise RuntimeError(f"Falha definitiva em {url}: {last_err}")


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
            rows.extend(csv.DictReader(lines, delimiter="\t"))
        match = NEXT_LINK.search(headers.get("Link", "") or headers.get("link", ""))
        url = match.group(1) if match else None
    return rows


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
# Seleção
# ---------------------------------------------------------------------------

def read_gene_list(path: str) -> list[str]:
    genes = []
    with open(path) as fh:
        for line in fh:
            gene = line.split("#")[0].strip()
            if gene and gene not in genes:
                genes.append(gene)
    return genes


def species_rank(taxid: int) -> int:
    return SPECIES_PRIORITY.index(taxid) if taxid in SPECIES_PRIORITY else len(SPECIES_PRIORITY)


def accepted_names(gene: str) -> set[str]:
    names = {gene.upper()}
    if gene in GENE_ALIASES:
        names.add(GENE_ALIASES[gene].upper())
    return names


def select_sequences(genes: list[str], taxon: int, max_seqs: int) -> tuple[dict, dict]:
    """
    Retorna (selecionadas, relatorio):
      selecionadas[gene] = [{accession, taxid, organismo, length}, ...]
      relatorio[gene]    = {status, descartados_simbolo, descartados_residuos, ...}
    """
    selected, report = {}, {}
    for i in range(0, len(genes), UNIPROT_GENE_BATCH):
        batch = genes[i:i + UNIPROT_GENE_BATCH]
        terms = sorted({n for g in batch for n in accepted_names(g)})
        gene_q = " OR ".join(f'gene_exact:"{t}"' for t in terms)
        query = (f"({gene_q}) AND (reviewed:true) AND (fragment:false) "
                 f"AND (taxonomy_id:{taxon})")
        rows = uniprot_tsv(query, "accession,gene_primary,organism_id,organism_name,length,sequence")

        name_to_gene = {n: g for g in batch for n in accepted_names(g)}
        candidates = {g: [] for g in batch}
        symbol_drops, residue_drops, length_drops = [], {g: [] for g in batch}, {g: [] for g in batch}
        for r in rows:
            primary = (r["Gene Names (primary)"] or "").strip()
            gene = name_to_gene.get(primary.upper())
            if gene is None:
                symbol_drops.append(f"{r['Entry']} ({primary or 'sem nome primário'}, {r['Organism']})")
                continue
            length = int(r["Length"])
            if not MIN_LENGTH <= length <= MAX_LENGTH:
                length_drops[gene].append(f"{r['Entry']} ({length} aa)")
                continue
            if INVALID_RESIDUES & set(r["Sequence"]):
                residue_drops[gene].append(r["Entry"])
                continue
            candidates[gene].append({
                "accession": r["Entry"], "taxid": int(r["Organism (ID)"]),
                "organismo": r["Organism"], "length": length, "nome_primario": primary,
            })
        if symbol_drops:
            log.info("  Descartados por símbolo (sinônimo ≠ gene pedido): %s", "; ".join(symbol_drops))

        for gene in batch:
            entries = sorted(candidates[gene], key=lambda e: (species_rank(e["taxid"]), e["accession"]))
            chosen, seen_tax = [], set()
            for e in entries:
                if e["taxid"] in seen_tax:
                    continue
                chosen.append(e)
                seen_tax.add(e["taxid"])
                if len(chosen) == max_seqs:
                    break
            has_human = any(e["taxid"] == HUMAN_TAXID for e in chosen)
            if has_human:
                status = "ok"
            elif chosen:
                status = "sem_humano"
            elif length_drops[gene] or residue_drops[gene]:
                status = "descartado_tamanho_ou_residuo"
            else:
                status = "nao_encontrado"
            report[gene] = {
                "status": status,
                "n_seqs": len(chosen) if has_human else 0,
                "nome_primario": chosen[0]["nome_primario"] if chosen else None,
                "organismos": [e["organismo"] for e in chosen] if has_human else [],
                "descartados_tamanho": length_drops[gene],
                "descartados_residuos": residue_drops[gene],
            }
            if has_human:
                selected[gene] = chosen
            else:
                log.warning("  %s: %s", gene, status)
        report.setdefault("_descartados_simbolo", []).extend(symbol_drops)
        log.info("  UniProt: %d/%d genes processados", min(i + UNIPROT_GENE_BATCH, len(genes)), len(genes))
    return selected, report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Coleta sequências Swiss-Prot e homólogos por gene")
    parser.add_argument("--genes", required=True, help="Arquivo com um símbolo por linha (# = comentário)")
    parser.add_argument("--out-dir", required=True, help="Diretório de saída dos FASTAs (nunca data/raw)")
    parser.add_argument("--max-seqs", type=int, default=MAX_SEQS_PER_GENE, help="Teto de sequências por gene")
    parser.add_argument("--taxon", type=int, default=MAMMALIA_TAXID, help="Taxon NCBI (default: Mammalia)")
    parser.add_argument("--overwrite", action="store_true", help="Refaz genes cujo FASTA já existe")
    args = parser.parse_args()

    if os.path.normpath(args.out_dir) == os.path.normpath(FROZEN_DIR):
        log.error(f"{FROZEN_DIR} é o dataset congelado do TCC I. Use outro --out-dir (ex.: data/raw_v2).")
        sys.exit(1)
    if not os.path.exists(args.genes):
        log.error(f"{args.genes} não encontrado. Rode antes: python src/00_build_gene_lists.py")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    genes = read_gene_list(args.genes)
    pending = [g for g in genes
               if args.overwrite or not os.path.exists(os.path.join(args.out_dir, f"{g}.fasta"))]
    log.info(f"{len(genes)} genes na lista; {len(pending)} a coletar "
             f"(taxon={args.taxon}, máx. {args.max_seqs}/gene) → {args.out_dir}")

    selected, report = select_sequences(pending, args.taxon, args.max_seqs)
    records = fetch_fasta([e["accession"] for entries in selected.values() for e in entries])

    for gene, entries in selected.items():
        with open(os.path.join(args.out_dir, f"{gene}.fasta"), "w") as fh:
            fh.writelines(records[e["accession"]] for e in entries if e["accession"] in records)

    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, f"{os.path.basename(os.path.normpath(args.out_dir))}.json")
    previous = {}
    if os.path.exists(report_path) and not args.overwrite:
        with open(report_path) as fh:
            previous = json.load(fh).get("genes", {})
    symbol_drops = report.pop("_descartados_simbolo", [])
    summary = {
        "lista": args.genes, "out_dir": args.out_dir, "taxon": args.taxon,
        "max_seqs": args.max_seqs, "min_length": MIN_LENGTH, "max_length": MAX_LENGTH,
        "residuos_excluidos": sorted(INVALID_RESIDUES), "aliases": GENE_ALIASES,
        "descartados_simbolo": symbol_drops,
        "genes": {**previous, **report},
    }
    with open(report_path, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    ok = [g for g, r in summary["genes"].items() if r["status"] == "ok"]
    n_seqs = sum(r["n_seqs"] for r in summary["genes"].values())
    failed = [g for g, r in summary["genes"].items() if r["status"] != "ok"]
    log.info(f"Concluído: {len(ok)} genes, {n_seqs} sequências. Relatório: {report_path}")
    if failed:
        log.warning(f"Sem sequência humana válida ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()
