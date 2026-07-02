"""Draw the 10-article pilot sample (docs/DECISIONS.md #2) from the local
corpus index built by src/build_corpus_index.py. No network calls.

Usage:
    python -m src.sample_articles --index data/corpus_index.csv --out data/sampled_articles.csv
"""
from __future__ import annotations

import argparse
import csv
import random

from src.subfields import PSYCHOLOGY_SUBFIELDS

CSV_FIELDS = [
    "doi",
    "subfield",
    "title",
    "publication_date",
    "year",
    "stage",
    "lead_institution",
    "subject_level_1",
    "subject",
    "xml_path",
]


def load_index(index_csv: str) -> list[dict]:
    with open(index_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stratified_sample(
    rows: list[dict],
    subfields: list[str] = PSYCHOLOGY_SUBFIELDS,
    per_subfield: int = 2,
    seed: int = 42,
) -> list[dict]:
    """Randomly select `per_subfield` articles from each subfield.

    `rows` is a list of dicts as produced by build_corpus_index.py (each
    with a `matched_subfields` field of ';'-joined subfield names) — passed
    in directly rather than read from disk here, so this is trivially
    unit-testable with fixture rows (tests/test_sample_articles.py).
    """
    rng = random.Random(seed)
    sampled: list[dict] = []
    for subfield in subfields:
        candidates = [r for r in rows if subfield in r.get("matched_subfields", "").split(";")]
        if len(candidates) < per_subfield:
            raise ValueError(
                f"Only {len(candidates)} candidates found for {subfield!r}, "
                f"need at least {per_subfield}. Has src/build_corpus_index.py run yet?"
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
            writer.writerow(article)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="data/corpus_index.csv")
    parser.add_argument("--out", default="data/sampled_articles.csv")
    parser.add_argument("--per-subfield", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_index(args.index)
    articles = stratified_sample(rows, per_subfield=args.per_subfield, seed=args.seed)
    write_csv(articles, args.out)
    print(f"Wrote {len(articles)} articles to {args.out}")


if __name__ == "__main__":
    main()
