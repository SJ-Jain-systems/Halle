"""Validate that PLOS Solr can enumerate psychology PLOS ONE research articles
directly (the fix for the allofplos XML taxonomy gaps, esp. 2015). Prints the
result count for a couple of years and a sample doc, so we can confirm the
query shape before building the real Solr-based indexer.

    python scripts/test_solr.py
"""
from __future__ import annotations

import json

import requests

SOLR_URL = "https://api.plos.org/search"


def count_psych(year: int) -> dict:
    q = (
        'journal:"PLOS ONE" AND article_type:"Research Article" '
        'AND subject:"Psychology" '
        f"AND publication_date:[{year}-01-01T00:00:00Z TO {year}-12-31T23:59:59Z]"
    )
    params = {"q": q, "fl": "id,publication_date,subject", "wt": "json", "rows": 1}
    r = requests.get(SOLR_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()["response"]


def main() -> None:
    for year in (2013, 2014, 2015, 2016, 2020):
        resp = count_psych(year)
        print(f"{year}: numFound = {resp['numFound']}")
    # Show one full 2015 doc to confirm subject paths are present.
    resp = count_psych(2015)
    if resp["docs"]:
        print("\nsample 2015 psychology doc:")
        print(json.dumps(resp["docs"][0], indent=2)[:1500])


if __name__ == "__main__":
    main()
