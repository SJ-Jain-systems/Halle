"""PLOS Solr access for article discovery (docs/DECISIONS.md #1, re-revised).

The allofplos XML is missing the subject taxonomy for a large share of some
years (≈99% of 2015, ≈37% of 2013) — see scripts/diagnose_2015.py. PLOS's
Solr index has the authoritative taxonomy for every article, so we use it to
enumerate the psychology article set (discovery) and keep the local allofplos
XML purely for full-text extraction. This is the "Solr for discovery,
allofplos for full text" split the brief originally described.

Query notes (learned the hard way — see scripts/test_solr.py):
  - `doc_type:full` is mandatory, else Solr returns sub-documents (millions).
  - Constraints must be filter queries (fq); a loose space-separated q is
    parsed permissively and matches ~everything.
  - We page per-year with start/rows (each year is a few thousand results, so
    start never exceeds Solr's deep-paging limit) rather than relying on
    cursorMark support.
"""
from __future__ import annotations

import os
import time
from typing import Iterator

import requests

SOLR_URL = "https://api.plos.org/search"


def _filter_queries(year: int) -> list[str]:
    return [
        "doc_type:full",
        'journal:"PLOS ONE"',
        'article_type:"Research Article"',
        f"publication_date:[{year}-01-01T00:00:00Z TO {year}-12-31T23:59:59Z]",
        'subject:"Psychology"',
    ]


def fetch_year(
    year: int,
    rows: int = 500,
    pause: float = 1.0,
    session: requests.Session | None = None,
    timeout: float = 60.0,
) -> Iterator[dict]:
    """Yield every psychology PLOS ONE research-article doc for one year."""
    session = session or requests.Session()
    start = 0
    while True:
        params = {
            "q": "*:*",
            "fq": _filter_queries(year),
            "fl": "id,subject,publication_date",
            "wt": "json",
            "rows": rows,
            "start": start,
        }
        resp = session.get(SOLR_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()["response"]
        docs = body.get("docs", [])
        for doc in docs:
            yield doc
        start += rows
        if start >= body.get("numFound", 0) or not docs:
            break
        time.sleep(pause)


def iter_psychology_articles(start_year: int, end_year: int, **kwargs) -> Iterator[dict]:
    session = requests.Session()
    for year in range(start_year, end_year + 1):
        yield from fetch_year(year, session=session, **kwargs)


def psychology_subfields_from_paths(subject_paths: list[str]) -> list[str]:
    """Extract the psychology subfield term(s) from Solr subject paths.

    Solr returns slash-delimited paths like
    "/Biology and life sciences/Psychology/Cognitive psychology/Decision making".
    The subfield is the segment immediately after a "Psychology" node
    ("Cognitive psychology"), matching jats_xml.get_psychology_subfields'
    semantics. A path where "Psychology" is the leaf yields "Psychology".
    Paths with no standalone "Psychology" segment (e.g. Neuroscience >
    Cognitive science > "Cognitive psychology") contribute nothing here — a
    genuine psychology article also carries a "/Psychology/..." path. Returns
    deduplicated terms in first-seen order; empty if not under Psychology.
    """
    subfields: list[str] = []
    for path in subject_paths:
        parts = [p for p in path.split("/") if p]
        if "Psychology" not in parts:
            continue
        i = parts.index("Psychology")
        term = parts[i + 1] if i + 1 < len(parts) else "Psychology"
        if term not in subfields:
            subfields.append(term)
    return subfields


def doi_to_xml_path(doi: str, corpus_dir: str) -> str:
    """Map a PLOS DOI to its local allofplos XML file.

    10.1371/journal.pone.0116314 -> <corpus_dir>/journal.pone.0116314.xml
    """
    basename = doi.split("/")[-1]
    return os.path.join(corpus_dir, f"{basename}.xml")
