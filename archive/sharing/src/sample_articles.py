# NOTES
# Picks the small test batch (the pilot) I use to check the model before
# running it on all 58,000 papers. It reads the master list and draws 2 papers
# from each subfield, at random but with a fixed seed so the same batch comes out
# every time.
#
# Why 2 per subfield and not 46 at random: a plain random draw would be swamped
# by the big subfields and might never test a rare one. This way the batch spans
# the whole range.
#
# Before drawing, I throw out any paper I don't have the full-text file for
# (about 6%, mostly very recent), so the batch never contains a paper the model
# can't read.
"""Draw the stratified pilot sample from the master list.

Usage:
    python -m src.sample_articles --index data/corpus_index.csv --out data/sampled_articles.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import random

logger = logging.getLogger(__name__)

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


def distinct_subfields(rows: list[dict]) -> list[str]:
    """List every subfield that shows up in the master list."""
    seen: list[str] = []
    for r in rows:
        for sf in r.get("matched_subfields", "").split(";"):
            sf = sf.strip()
            if sf and sf not in seen:
                seen.append(sf)
    return seen


def stratified_sample(
    rows: list[dict],
    subfields: list[str] | None = None,
    per_subfield: int = 2,
    seed: int = 42,
) -> list[dict]:
    """Pick per_subfield papers from each subfield, at random but seeded.

    For each subfield, gather every paper tagged with it and pick 2.
    random.Random(seed) is a fixed dice roll, so the picks are reproducible. If a
    subfield has fewer than 2 papers I skip it rather than crash (in auto mode).

    Passing rows in directly, rather than reading a file here, keeps this easy to
    unit-test.
    """
    rng = random.Random(seed)
    auto = subfields is None
    if auto:
        subfields = distinct_subfields(rows)
    sampled: list[dict] = []
    for subfield in subfields:
        candidates = [
            r for r in rows
            if subfield in [s.strip() for s in r.get("matched_subfields", "").split(";")]
        ]
        if len(candidates) < per_subfield:
            if auto:
                logger.warning(
                    "Skipping subfield %r: only %d article(s), need %d",
                    subfield, len(candidates), per_subfield,
                )
                continue
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


def filter_to_local_xml(rows: list[dict]) -> list[dict]:
    """Keep only papers whose full-text file actually exists on disk.

    About 6% of the index is papers, mostly very recent, that aren't in the local
    snapshot. I can't read those, so they must not go into the pilot.
    """
    kept = [r for r in rows if r.get("xml_path") and os.path.exists(r["xml_path"])]
    dropped = len(rows) - len(kept)
    if dropped:
        logger.info("Dropped %d/%d index rows with no local XML file", dropped, len(rows))
    return kept


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="data/corpus_index.csv")
    parser.add_argument("--out", default="data/sampled_articles.csv")
    parser.add_argument("--per-subfield", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load the master list, drop unreadable papers, draw the sample, write it out,
    # and report what I covered.
    rows = filter_to_local_xml(load_index(args.index))
    articles = stratified_sample(rows, per_subfield=args.per_subfield, seed=args.seed)
    write_csv(articles, args.out)
    subfields_covered = sorted({a["subfield"] for a in articles})
    print(f"Wrote {len(articles)} articles to {args.out}")
    print(f"Covered {len(subfields_covered)} subfields: {', '.join(subfields_covered)}")


if __name__ == "__main__":
    main()
