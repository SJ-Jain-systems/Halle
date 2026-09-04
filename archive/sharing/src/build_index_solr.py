# NOTES
# This is the script I run to build the master list of papers
# (data/corpus_index.csv, about 58,000 rows). It pulls every psychology paper
# from the search engine, works out each one's subfield and time stage, points
# at where its full-text file lives, and writes one row per paper.
#
# Run it on the login node. It needs internet but it's light. About 10 to 20
# minutes.
#
# It replaced an earlier version that scanned the downloaded files directly. I
# dropped that because the files are missing tags for 2015 and part of 2013.
#
# If the search engine returns a paper but I can't read a subfield out of its
# tags, I skip it, so the list stays consistent with how I define psychology
# everywhere else.
"""Build the master list of psychology papers from the PLOS search index.

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

# Columns of the master list. One row is one paper.
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
    # Solr dates look like "2015-02-25T00:00:00Z". I only want the date part,
    # so keep the first 10 characters.
    try:
        return dt.date.fromisoformat(value[:10])
    except (ValueError, TypeError):
        return None


def build_index(out_path: str, corpus_dir: str, start_year: int = MIN_YEAR,
                end_year: int = MAX_YEAR, log_every: int = 1000) -> int:
    # Four counters so the final log line says exactly what happened. seen is how
    # many the engine returned. kept is how many I wrote. no_subfield is how
    # many I dropped for having no readable subfield. missing_xml is how many I
    # kept but don't have the full-text file for yet.
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
                # Solr matched on Psychology but the paths have no real
                # Psychology node. Skip, to match how I define it elsewhere.
                no_subfield += 1
                continue
            doi = doc["id"]
            pub_date = _parse_date(doc.get("publication_date", ""))
            xml_path = doi_to_xml_path(doi, corpus_dir)
            if not os.path.exists(xml_path):
                # Keep the paper, but note I can't read it yet.
                missing_xml += 1
            writer.writerow({
                "doi": doi,
                "publication_date": pub_date.isoformat() if pub_date else "",
                "year": pub_date.year if pub_date else "",
                "stage": stage_for_date(pub_date) if pub_date else "",
                "matched_subfields": ";".join(subfields),
                "subject": ";".join(subjects),
                "lead_institution": "",  # filled from the local file later if needed
                "xml_path": xml_path,
            })
            kept += 1
            if seen % log_every == 0:
                # Print progress and flush so I can watch it live.
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
