# NOTES
# This is how we find the psychology papers. It asks PLOS's search engine
# (Solr) for every PLOS ONE research article tagged as psychology, one year at
# a time.
#
# We use the search engine instead of the downloaded files because the files
# are missing their subject tags for whole years. 2015 is basically empty in
# the files. The search engine has the tags for every article.
#
# Nothing here reads full text. It only builds the list of which papers to read.
"""Find psychology papers through the PLOS Solr search index.

Two things about the query, both learned by trial and error:
- You must pass doc_type:full. Without it Solr returns page fragments, millions
  of them, not articles.
- The conditions have to be filter queries (fq). A plain query gets read loosely
  and matches almost everything.
"""
from __future__ import annotations

import os
import time
from typing import Iterator

import requests

SOLR_URL = "https://api.plos.org/search"


def _filter_queries(year: int) -> list[str]:
    # These five conditions are the whole definition of a paper we want.
    # doc_type:full is not optional. Drop it and you get millions of fragments.
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
    """Return every psychology PLOS ONE research article for one year."""
    # The engine won't hand back thousands at once. We ask in pages of 500 and
    # move "start" forward until we've seen them all. The pause is politeness so
    # we don't hammer PLOS.
    session = session or requests.Session()
    start = 0
    while True:
        params = {
            "q": "*:*",
            "fq": _filter_queries(year),
            "fl": "id,subject,publication_date",  # only the 3 fields we use
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
        # Stop once we've paged past the total, or a page comes back empty.
        if start >= body.get("numFound", 0) or not docs:
            break
        time.sleep(pause)


def iter_psychology_articles(start_year: int, end_year: int, **kwargs) -> Iterator[dict]:
    # We go year by year on purpose. Each year is only a few thousand results,
    # which keeps paging simple and well inside the engine's limits.
    session = requests.Session()
    for year in range(start_year, end_year + 1):
        yield from fetch_year(year, session=session, **kwargs)


def psychology_subfields_from_paths(subject_paths: list[str]) -> list[str]:
    """Pull the psychology subfield out of the tag paths Solr returns.

    Solr gives each tag as a full path, like
    "/Biology and life sciences/Psychology/Cognitive psychology/...".
    We split on the slashes, find "Psychology", and take the next piece. That
    piece is the subfield. One paper often carries the same subfield twice
    (filed under both Biology and Social sciences), so we de-duplicate.
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
    """Work out where a paper's downloaded file sits, from its DOI.

    10.1371/journal.pone.0116314  ->  <corpus_dir>/journal.pone.0116314.xml
    """
    # The last chunk of the DOI is literally the filename. Take the part after
    # the final slash and add .xml.
    basename = doi.split("/")[-1]
    return os.path.join(corpus_dir, f"{basename}.xml")
