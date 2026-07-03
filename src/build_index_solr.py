"""Build the filtered corpus index from PLOS Solr (authoritative, gap-free),
replacing the XML-scan index that missed taxonomy-less articles (esp. 2015).

Runs on a node with internet (the Rivanna login node) — it's network-bound,
not compute-bound, so no sbatch needed. Enumerates every psychology PLOS ONE
research article 2010–2026 from Solr, parses the subfield(s) from the subject
paths, and maps each DOI to its local allofplos XML file (for later full-text
extraction). Lead-author institution is left blank here and can be enriched
from the local XML in a separate CPU step if needed.

Usage:
    python -m src.build_index_solr --out data/corpus_index.csv --corpus-dir "$PLOS_CORPUS"
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import os

from src.allofplos_client import DEFAULT_CORPUS_DIR
from src.solr_client import (
    doi_to_xml_path,
    iter_psychology_articles,
    psychology_subfields_from_paths,
)
from src.subfields import stage_for_date

logger = logging.getLogger(__name__)

MIN_YEAR = 2010
MAX_YEAR = 2026

CSV_FIELDS = [
    "doi",
    "publication_date",
    "year",
    "stage",
    "matched_subfields",
    "subject",
    "lead_institution",
    "xml_path",
]


def _parse_date(value: str) -> dt.date | None:
    # Solr publication_date looks like "2015-02-25T00:00:00Z".
    try:
        return dt.date.fromisoformat(value[:10])
    except (ValueError, TypeError):
        return None


def build_index(out_path: str, corpus_dir: str, start_year: int = MIN_YEAR,
                end_year: int = MAX_YEAR, log_every: int = 1000) -> int:
    kept = 0
    seen = 0
    no_subfield = 0
    missing_xml = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for doc in iter_psychology_articles(start_year, end_year):
            seen += 1
            subjects = doc.get("subject", []) or []
            subfields = psychology_subfields_from_paths(subjects)
            if not subfields:
                # Solr matched subject:"Psychology" but no standalone Psychology
                # node in the paths — keep semantics consistent with the XML
                # parser and skip these.
                no_subfield += 1
                continue
            doi = doc["id"]
            pub_date = _parse_date(doc.get("publication_date", ""))
            xml_path = doi_to_xml_path(doi, corpus_dir)
            if not os.path.exists(xml_path):
                missing_xml += 1
            writer.writerow({
                "doi": doi,
                "publication_date": pub_date.isoformat() if pub_date else "",
                "year": pub_date.year if pub_date else "",
                "stage": stage_for_date(pub_date) if pub_date else "",
                "matched_subfields": ";".join(subfields),
                "subject": ";".join(subjects),
                "lead_institution": "",
                "xml_path": xml_path,
            })
            kept += 1
            if seen % log_every == 0:
                logger.info("Fetched %d articles, kept %d", seen, kept)
                f.flush()
    logger.info(
        "Done: %d Solr hits, kept %d (skipped %d with no Psychology-node subfield); "
        "%d kept articles have no local XML file",
        seen, kept, no_subfield, missing_xml,
    )
    return kept


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/corpus_index.csv")
    parser.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--start-year", type=int, default=MIN_YEAR)
    parser.add_argument("--end-year", type=int, default=MAX_YEAR)
    args = parser.parse_args()
    build_index(args.out, args.corpus_dir, args.start_year, args.end_year)


if __name__ == "__main__":
    main()
