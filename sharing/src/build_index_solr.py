# ============================================================================
# PLAIN-ENGLISH NOTES (for colleagues reading this file)
#
# What this file is for: this is the script we actually run to produce the
# master list of papers (data/corpus_index.csv, ~58,000 rows). It uses
# solr_client.py to pull every psychology paper from PLOS's search engine,
# works out each paper's subfield and time-stage, points at where its full-text
# file lives on disk, and writes one row per paper to a spreadsheet.
#
# Run it on the login node (it needs internet). It is not heavy - it just talks
# to the search engine and writes a file, ~10-20 minutes. This REPLACED an
# earlier version that scanned the downloaded files directly, which we ditched
# because those files are missing tags for 2015 and part of 2013.
#
# One deliberate choice: if the search engine returns a paper but we can't find
# its subfield in the tag paths, we skip it, so the list stays consistent with
# how we define "psychology" everywhere else.
# ============================================================================

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

# The columns of the master list. One row = one paper.
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
    # We only want the date part, so chop off everything after the first 10 chars.
    try:
        return dt.date.fromisoformat(value[:10])
    except (ValueError, TypeError):
        return None


def build_index(out_path: str, corpus_dir: str, start_year: int = MIN_YEAR,
                end_year: int = MAX_YEAR, log_every: int = 1000) -> int:
    # Counters so the final log line tells us exactly what happened:
    # how many the search engine returned (seen), how many we wrote (kept),
    # how many we dropped for having no readable subfield (no_subfield), and
    # how many we kept but don't have the full-text file for yet (missing_xml).
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
                # We still record the paper, but note we can't read it yet.
                missing_xml += 1
            writer.writerow({
                "doi": doi,
                "publication_date": pub_date.isoformat() if pub_date else "",
                "year": pub_date.year if pub_date else "",
                "stage": stage_for_date(pub_date) if pub_date else "",
                "matched_subfields": ";".join(subfields),
                "subject": ";".join(subjects),
                "lead_institution": "",  # filled in later from the local file if needed
                "xml_path": xml_path,
            })
            kept += 1
            if seen % log_every == 0:
                # Print progress and flush to disk so we can watch it live.
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
