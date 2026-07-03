"""Scan the local allofplos corpus and write the filtered population index
(brief item 3, Inclusion Criteria) — the master list every later stage
(sampling, full-scale extraction) reads from. Replaces what a Solr query
would have done, entirely offline.

Usage:
    python -m src.build_corpus_index --corpus-dir ~/allofplos_corpus --out data/corpus_index.csv

Filters applied (brief item 3):
  - journal is PLOS ONE
  - article-type is "research-article" (empirical article)
  - tagged with at least one of the 5 target psychology subfields
  - publication year in [2010, 2026]
No sample-size filter is applied — the brief explicitly says "all sample
sizes (final sample)".
"""
from __future__ import annotations

import argparse
import csv
import logging

from src.allofplos_client import DEFAULT_CORPUS_DIR, iter_corpus_xml
from src.jats_xml import ArticleMetadata, parse_metadata
from src.subfields import matched_subfields, stage_for_date

logger = logging.getLogger(__name__)

TARGET_JOURNAL_SUBSTRING = "plos one"
TARGET_ARTICLE_TYPE = "research-article"
MIN_YEAR = 2010
MAX_YEAR = 2026

CSV_FIELDS = [
    "doi",
    "title",
    "journal",
    "article_type",
    "publication_date",
    "year",
    "stage",
    "matched_subfields",
    "subject_level_1",
    "subject",
    "lead_institution",
    "xml_path",
]


def passes_inclusion_criteria(meta: ArticleMetadata) -> tuple[bool, list[str]]:
    if TARGET_JOURNAL_SUBSTRING not in meta.journal.lower():
        return False, []
    if meta.article_type != TARGET_ARTICLE_TYPE:
        return False, []
    if meta.publication_date is None or not (MIN_YEAR <= meta.publication_date.year <= MAX_YEAR):
        return False, []
    subfields = matched_subfields(meta.subject)
    if not subfields:
        return False, []
    return True, subfields


def to_row(meta: ArticleMetadata, subfields: list[str]) -> dict:
    return {
        "doi": meta.doi,
        "title": meta.title,
        "journal": meta.journal,
        "article_type": meta.article_type,
        "publication_date": meta.publication_date.isoformat() if meta.publication_date else "",
        "year": meta.publication_date.year if meta.publication_date else "",
        "stage": stage_for_date(meta.publication_date) if meta.publication_date else "",
        "matched_subfields": ";".join(subfields),
        "subject_level_1": ";".join(meta.subject_level_1),
        "subject": ";".join(meta.subject),
        "lead_institution": meta.lead_institution or "",
        "xml_path": meta.xml_path,
    }


def build_index(corpus_dir: str, out_path: str, log_every: int = 5000) -> int:
    kept = 0
    scanned = 0
    # Materialize the file list up front so we can log a total and give a
    # meaningful "scanned N/TOTAL" progress readout with an implied ETA.
    xml_paths = list(iter_corpus_xml(corpus_dir))
    total = len(xml_paths)
    logger.info("Found %d article XML files under %s; scanning...", total, corpus_dir)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for xml_path in xml_paths:
            scanned += 1
            try:
                meta = parse_metadata(xml_path)
            except Exception:
                logger.warning("Failed to parse %s, skipping", xml_path, exc_info=True)
                continue
            ok, subfields = passes_inclusion_criteria(meta)
            if ok:
                writer.writerow(to_row(meta, subfields))
                kept += 1
            if scanned % log_every == 0:
                # Flush both the log and the CSV so `tail`/`wc -l` show live
                # progress instead of sitting empty behind block buffering.
                logger.info("Scanned %d/%d articles, kept %d so far", scanned, total, kept)
                f.flush()
    logger.info("Done: scanned %d articles, kept %d matching inclusion criteria", scanned, kept)
    return kept


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--out", default="data/corpus_index.csv")
    args = parser.parse_args()
    build_index(args.corpus_dir, args.out)


if __name__ == "__main__":
    main()
