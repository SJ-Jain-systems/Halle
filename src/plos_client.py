"""Data access layer implementing docs/DECISIONS.md #1: Solr for discovery,
allofplos for full-text retrieval.

Network note: this module has not been exercised against the live PLOS API —
see docs/DECISIONS.md for why. The query shapes follow the public Solr
examples at https://api.plos.org/solr/examples/; re-verify field names there
if PLOS changes their schema.
"""
from __future__ import annotations

import requests

SOLR_SELECT_URL = "https://api.plos.org/solr/select"
DEFAULT_JOURNAL_KEY = "PLoSONE"
DEFAULT_ARTICLE_TYPE = "Research Article"
SOLR_FIELDS = "id,title,publication_date,author_display,subject,subject_level_1,journal"


class PlosApiError(RuntimeError):
    """Raised when the Solr endpoint returns a non-2xx response."""


def _build_query(subject_level_1_term: str, journal_key: str, article_type: str,
                  start_year: int | None, end_year: int | None) -> str:
    clauses = [
        f'subject_level_1:"{subject_level_1_term}"',
        f'cross_published_journal_key:{journal_key}',
        f'article_type:"{article_type}"',
    ]
    if start_year and end_year:
        clauses.append(
            f"publication_date:[{start_year}-01-01T00:00:00Z TO {end_year}-12-31T23:59:59Z]"
        )
    return " AND ".join(clauses)


def search_articles(
    subject_level_1_term: str,
    journal_key: str = DEFAULT_JOURNAL_KEY,
    article_type: str = DEFAULT_ARTICLE_TYPE,
    start_year: int | None = None,
    end_year: int | None = None,
    rows: int = 100,
    start: int = 0,
    timeout: float = 15.0,
) -> list[dict]:
    """Query the PLOS Solr endpoint for a single psychology subfield.

    Returns a list of docs with fields from SOLR_FIELDS. The `id` field is
    the article DOI for PLOS content.
    """
    params = {
        "q": _build_query(subject_level_1_term, journal_key, article_type, start_year, end_year),
        "fl": SOLR_FIELDS,
        "wt": "json",
        "rows": rows,
        "start": start,
    }
    response = requests.get(SOLR_SELECT_URL, params=params, timeout=timeout)
    if not response.ok:
        raise PlosApiError(f"Solr query failed ({response.status_code}): {response.text[:500]}")
    payload = response.json()
    return payload.get("response", {}).get("docs", [])


def fetch_fulltext(doi: str) -> str:
    """Fetch full-text JATS XML for a DOI via allofplos.

    Requires the `allofplos` package (see requirements.txt). allofplos
    resolves a DOI against its local corpus mirror, syncing from
    github.com/PLOS/allofplos on first use if the article isn't cached yet.
    """
    from allofplos.article import Article  # deferred import: heavy, network-syncing dependency

    article = Article(doi)
    return article.xml
