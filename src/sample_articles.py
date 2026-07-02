"""Draw the 10-article pilot sample (docs/DECISIONS.md #2).

Usage:
    python -m src.sample_articles --out data/sampled_articles.csv

Requires network access to api.plos.org (blocked in this sandbox — see
docs/DECISIONS.md). Run from an environment that has it.
"""
from __future__ import annotations

import argparse
import csv
import random
from typing import Callable

from src.plos_client import search_articles
from src.subfields import PSYCHOLOGY_SUBFIELDS

FetchFn = Callable[[str], list[dict]]

CSV_FIELDS = ["doi", "subfield", "title", "publication_date", "author_display", "subject_level_1"]


def stratified_sample(
    subfields: list[str] = PSYCHOLOGY_SUBFIELDS,
    per_subfield: int = 2,
    seed: int = 42,
    fetch_fn: FetchFn = lambda subfield: search_articles(subfield, rows=100),
) -> list[dict]:
    """Randomly select `per_subfield` articles from each subfield.

    `fetch_fn` is injectable so this can be unit-tested without hitting the
    live Solr endpoint (see tests/test_sample_articles.py).
    """
    rng = random.Random(seed)
    sampled: list[dict] = []
    for subfield in subfields:
        candidates = fetch_fn(subfield)
        if len(candidates) < per_subfield:
            raise ValueError(
                f"Only {len(candidates)} candidates found for {subfield!r}, "
                f"need at least {per_subfield}"
            )
        for doc in rng.sample(candidates, per_subfield):
            row = dict(doc)
            row["subfield"] = subfield
            sampled.append(row)
    return sampled


def write_csv(articles: list[dict], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for article in articles:
            row = dict(article)
            row["doi"] = row.get("id", row.get("doi", ""))
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/sampled_articles.csv")
    parser.add_argument("--per-subfield", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    articles = stratified_sample(per_subfield=args.per_subfield, seed=args.seed)
    write_csv(articles, args.out)
    print(f"Wrote {len(articles)} articles to {args.out}")


if __name__ == "__main__":
    main()
