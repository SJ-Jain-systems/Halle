# ============================================================================
# PLAIN-ENGLISH NOTES (for colleagues reading this file)
#
# What this file is for: this is how we FIND the psychology papers. It talks to
# PLOS's public search engine (called Solr) and asks it, one year at a time,
# "give me every PLOS ONE research article tagged as psychology." We use the
# search engine instead of the downloaded files because the files are missing
# their subject tags for whole years (2015 especially) - the search engine has
# those tags for every article.
#
# The three functions that matter:
#   - fetch_year / iter_psychology_articles: do the actual asking, page by page.
#   - psychology_subfields_from_paths: read the subfield (e.g. "Social
#     psychology") out of the tag text the search engine hands back.
#   - doi_to_xml_path: given a paper's ID, work out where its downloaded file
#     lives on disk (so a later step can read the full text).
#
# Nothing here reads full text. It only builds the list of which papers to read.
# ============================================================================

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
    # These five conditions are the whole definition of "a paper we want."
    # doc_type:full is NOT optional - without it the search returns millions of
    # page fragments instead of actual articles. Learned that the hard way.
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
    # The search engine won't hand back thousands of results at once, so we ask
    # in pages of 500 ("start" moves forward each loop) until we've seen them
    # all. The pause between pages is basic politeness so we don't hammer PLOS.
    session = session or requests.Session()
    start = 0
    while True:
        params = {
            "q": "*:*",
            "fq": _filter_queries(year),
            "fl": "id,subject,publication_date",  # only pull the 3 fields we use
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
        # Stop once we've paged past the total count, or a page comes back empty.
        if start >= body.get("numFound", 0) or not docs:
            break
        time.sleep(pause)


def iter_psychology_articles(start_year: int, end_year: int, **kwargs) -> Iterator[dict]:
    # We loop year by year on purpose: each year is only a few thousand results,
    # which keeps the paging simple and well within the search engine's limits.
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
    # In plain terms: the search engine gives each tag as a full path like
    # "/Biology.../Psychology/Social psychology/...". We split on the slashes,
    # find the word "Psychology", and grab whatever comes right after it - that
    # is the subfield. One paper often carries the same subfield twice (filed
    # under both Biology and Social sciences), so we de-duplicate.
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
    # The DOI's last chunk is literally the filename of the downloaded article,
    # so this is just string surgery: take the bit after the final slash and add
    # ".xml". That's the file the full-text reader will open later.
    basename = doi.split("/")[-1]
    return os.path.join(corpus_dir, f"{basename}.xml")
