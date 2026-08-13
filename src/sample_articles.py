"""Draw the pilot sample (docs/DECISIONS.md #2 — 100 articles by default, per
the 7/15 meeting) from the local corpus index built by
src/build_corpus_index.py. No network calls.

The sample guarantees a floor of `min_per_subfield` articles from every
psychology subfield present (so no subfield is missed), then tops up to
`total_size` total by allocating the remaining slots proportionally to each
subfield's article count.

Usage:
    python -m src.sample_articles --index data/corpus_index.csv --out data/sampled_articles.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import random

import yaml

from src.subject_filter import non_human_subject_reason

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
    """All psychology subfields present in the index, in first-seen order."""
    seen: list[str] = []
    for r in rows:
        for sf in r.get("matched_subfields", "").split(";"):
            sf = sf.strip()
            if sf and sf not in seen:
                seen.append(sf)
    return seen


def _candidates_for(rows: list[dict], subfield: str) -> list[dict]:
    return [
        r for r in rows
        if subfield in [s.strip() for s in r.get("matched_subfields", "").split(";")]
    ]


def _largest_remainder_alloc(remaining: int, capacity: dict[str, int]) -> dict[str, int]:
    """Distribute `remaining` slots across subfields proportionally to
    `capacity` (spare articles available), capped by capacity, using
    largest-remainder rounding. Deterministic given the capacity dict order."""
    alloc = {sf: 0 for sf in capacity}
    total_cap = sum(capacity.values())
    remaining = min(remaining, total_cap)
    if remaining <= 0 or total_cap == 0:
        return alloc

    quotas = {sf: remaining * capacity[sf] / total_cap for sf in capacity}
    for sf in capacity:
        alloc[sf] = min(int(quotas[sf]), capacity[sf])
    leftover = remaining - sum(alloc.values())
    # Hand out the leftover by largest fractional remainder, skipping any
    # subfield already at capacity.
    by_remainder = sorted(capacity, key=lambda sf: quotas[sf] - int(quotas[sf]), reverse=True)
    while leftover > 0:
        progressed = False
        for sf in by_remainder:
            if leftover <= 0:
                break
            if alloc[sf] < capacity[sf]:
                alloc[sf] += 1
                leftover -= 1
                progressed = True
        if not progressed:
            break
    return alloc


def stratified_sample(
    rows: list[dict],
    subfields: list[str] | None = None,
    per_subfield: int = 2,
    seed: int = 42,
    total_size: int | None = None,
    min_per_subfield: int | None = None,
) -> list[dict]:
    """Stratified pilot draw over psychology subfields.

    `rows` is a list of dicts as produced by build_corpus_index.py (each with
    a `matched_subfields` field of ';'-joined subfield names) — passed in
    directly rather than read from disk here, so this is trivially
    unit-testable with fixture rows (tests/test_sample_articles.py).

    Two modes:

    - `total_size is None` (legacy): take exactly `per_subfield` from every
      eligible subfield — 2×N articles for N subfields.
    - `total_size` set: guarantee a floor of `min_per_subfield` (defaults to
      `per_subfield`) from every eligible subfield, then top up to `total_size`
      total by allocating the remaining slots proportionally to each subfield's
      article count (largest-remainder rounding). If the floors alone already
      exceed `total_size`, coverage wins and the floor sample is returned.

    When `subfields` is None (the default, used by the CLI), subfields are taken
    from whatever the index contains, and any subfield with fewer than the floor
    is skipped with a warning — keeping the pilot representative across *all*
    psychology subfields PLOS uses without a hardcoded list. When `subfields` is
    given explicitly, a subfield short on candidates is an error instead.
    """
    rng = random.Random(seed)
    auto = subfields is None
    if auto:
        subfields = distinct_subfields(rows)
    floor = min_per_subfield if min_per_subfield is not None else per_subfield

    # A deterministic shuffled candidate order per eligible subfield; we take
    # prefixes of these (floor first, then any top-up), so one shuffle drives
    # both draws and the result stays reproducible for a given seed.
    order: dict[str, list[dict]] = {}
    for subfield in subfields:
        candidates = _candidates_for(rows, subfield)
        if len(candidates) < floor:
            if auto:
                logger.warning(
                    "Skipping subfield %r: only %d article(s), need %d",
                    subfield, len(candidates), floor,
                )
                continue
            raise ValueError(
                f"Only {len(candidates)} candidates found for {subfield!r}, "
                f"need at least {floor}. Has src/build_corpus_index.py run yet?"
            )
        pool = list(candidates)
        rng.shuffle(pool)
        order[subfield] = pool

    taken = {sf: floor for sf in order}
    if total_size is not None:
        remaining = total_size - sum(taken.values())
        capacity = {sf: len(order[sf]) - floor for sf in order}
        for sf, extra in _largest_remainder_alloc(remaining, capacity).items():
            taken[sf] += extra

    sampled: list[dict] = []
    for subfield, pool in order.items():
        for doc in pool[: taken[subfield]]:
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


def filter_out_non_human(rows: list[dict]) -> list[dict]:
    """Drop any article whose subject taxonomy marks it as a non-human
    (animal-model / model-organism) study, so it can never be drawn into the
    manual-encoding sample or the LLM test set (src/subject_filter.py).

    build_corpus_index.py already applies this filter when writing the index,
    so on a freshly built index this drops nothing. It is repeated here as a
    safety net: sampling from an index built before this filter existed (or by
    hand) must still not surface animal studies to a coder."""
    kept = [r for r in rows if non_human_subject_reason(r.get("subject")) is None]
    dropped = len(rows) - len(kept)
    if dropped:
        logger.info("Dropped %d/%d index rows as non-human/animal studies", dropped, len(rows))
    return kept


def filter_to_local_xml(rows: list[dict]) -> list[dict]:
    """Drop rows whose xml_path isn't present on disk — ~6% of the Solr index
    is articles (mostly very recent) not in the local corpus snapshot, which
    can't be full-text extracted, so they must not be drawn into the pilot."""
    kept = [r for r in rows if r.get("xml_path") and os.path.exists(r["xml_path"])]
    dropped = len(rows) - len(kept)
    if dropped:
        logger.info("Dropped %d/%d index rows with no local XML file", dropped, len(rows))
    return kept


def _sampling_config(config_path: str) -> dict:
    """Pilot size / floor / seed defaults from config.yaml (validation +
    sampling blocks), falling back to hardcoded defaults if absent."""
    defaults = {"pilot_size": 100, "min_per_subfield": 2, "seed": 42}
    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except OSError:
        return defaults
    validation = config.get("validation") or {}
    sampling = config.get("sampling") or {}
    return {
        "pilot_size": validation.get("pilot_size", defaults["pilot_size"]),
        "min_per_subfield": validation.get("min_per_subfield", defaults["min_per_subfield"]),
        "seed": sampling.get("seed", defaults["seed"]),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="data/corpus_index.csv")
    parser.add_argument("--out", default="data/sampled_articles.csv")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--size", type=int, default=None,
                        help="Total pilot size (default: config validation.pilot_size)")
    parser.add_argument("--min-per-subfield", type=int, default=None,
                        help="Floor drawn from each subfield (default: config validation.min_per_subfield)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = _sampling_config(args.config)
    size = args.size if args.size is not None else cfg["pilot_size"]
    floor = args.min_per_subfield if args.min_per_subfield is not None else cfg["min_per_subfield"]
    seed = args.seed if args.seed is not None else cfg["seed"]

    rows = filter_out_non_human(filter_to_local_xml(load_index(args.index)))
    articles = stratified_sample(
        rows, per_subfield=floor, min_per_subfield=floor, total_size=size, seed=seed
    )
    write_csv(articles, args.out)
    subfields_covered = sorted({a["subfield"] for a in articles})
    print(f"Wrote {len(articles)} articles to {args.out} (target {size}, floor {floor}/subfield)")
    print(f"Covered {len(subfields_covered)} subfields: {', '.join(subfields_covered)}")


if __name__ == "__main__":
    main()
