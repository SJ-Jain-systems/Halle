"""Incrementally diagnose the PLOS Solr query: add one filter at a time for a
single year and print numFound after each, plus the params Solr echoes back.
Isolates which filter isn't biting (and checks the doc_type:full gotcha).

    python scripts/test_solr.py
"""
from __future__ import annotations

import json

import requests

SOLR_URL = "https://api.plos.org/search"
YEAR = 2015


def run(label: str, fq: list[str], rows: int = 0) -> None:
    params = {"q": "*:*", "fq": fq, "fl": "id,journal,article_type,publication_date", "wt": "json", "rows": rows}
    r = requests.get(SOLR_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    print(f"{label}: numFound = {data['response']['numFound']}")
    if rows and data["response"]["docs"]:
        print("   sample:", json.dumps(data["response"]["docs"][0])[:300])


def main() -> None:
    date = f"publication_date:[{YEAR}-01-01T00:00:00Z TO {YEAR}-12-31T23:59:59Z]"
    run("q=*:* only", [])
    run("doc_type:full", ["doc_type:full"])
    run("+ journal PLOS ONE", ["doc_type:full", 'journal:"PLOS ONE"'])
    run("+ article_type Research Article",
        ["doc_type:full", 'journal:"PLOS ONE"', 'article_type:"Research Article"'])
    run(f"+ date {YEAR}",
        ["doc_type:full", 'journal:"PLOS ONE"', 'article_type:"Research Article"', date])
    run("+ subject Psychology",
        ["doc_type:full", 'journal:"PLOS ONE"', 'article_type:"Research Article"', date,
         'subject:"Psychology"'], rows=1)


if __name__ == "__main__":
    main()
