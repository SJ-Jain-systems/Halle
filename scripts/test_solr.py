"""Validate that PLOS Solr can enumerate psychology PLOS ONE research articles
directly (the fix for the allofplos XML taxonomy gaps, esp. 2015).

Uses Solr filter queries (fq), which are strictly ANDed — a plain space-
separated q gets parsed loosely by PLOS's endpoint and returns ~everything.
Prints, per year, the total research-article count vs the Psychology subset,
so we can confirm the subject filter actually bites.

    python scripts/test_solr.py
"""
from __future__ import annotations

import json

import requests

SOLR_URL = "https://api.plos.org/search"


def query(year: int, with_psych: bool, rows: int = 1) -> dict:
    fq = [
        'journal:"PLOS ONE"',
        'article_type:"Research Article"',
        f"publication_date:[{year}-01-01T00:00:00Z TO {year}-12-31T23:59:59Z]",
    ]
    if with_psych:
        fq.append('subject:"Psychology"')
    params = {"q": "*:*", "fq": fq, "fl": "id,publication_date,subject", "wt": "json", "rows": rows}
    r = requests.get(SOLR_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()["response"]


def main() -> None:
    print("year | total research articles | with subject:Psychology")
    for year in (2013, 2014, 2015, 2016, 2020):
        total = query(year, with_psych=False)["numFound"]
        psych = query(year, with_psych=True)["numFound"]
        print(f"  {year} | {total:6d} | {psych:6d}")
    resp = query(2015, with_psych=True, rows=1)
    if resp["docs"]:
        print("\nsample 2015 psychology doc:")
        print(json.dumps(resp["docs"][0], indent=2)[:1200])


if __name__ == "__main__":
    main()
