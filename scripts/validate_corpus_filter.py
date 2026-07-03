"""Quick validation of the corpus filter against real allofplos XML, without
running the full (~1.5h) index build. Samples PLOS ONE articles spread across
the corpus, reports how many pass each inclusion criterion, and lists the
distinct psychology subfields found.

Run this after any change to the parsing/filter logic, before committing GPU
or long-CPU time to the full build:

    python scripts/validate_corpus_filter.py --corpus-dir "$PLOS_CORPUS"
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import Counter

# Allow running as `python scripts/validate_corpus_filter.py` (which otherwise
# puts scripts/ on sys.path, not the repo root) — add the repo root so `src`
# is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.build_corpus_index import (  # noqa: E402
    MAX_YEAR,
    MIN_YEAR,
    TARGET_ARTICLE_TYPE,
    TARGET_JOURNAL_SUBSTRING,
)
from src.jats_xml import parse_metadata  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", default=os.environ.get("PLOS_CORPUS", ""))
    parser.add_argument("--sample-size", type=int, default=2000)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.corpus_dir, "journal.pone.*.xml")))
    if not files:
        raise SystemExit(f"No PLOS ONE XML found under {args.corpus_dir!r}")
    step = max(1, len(files) // args.sample_size)
    sample = files[::step]
    print(f"sampling {len(sample)} files spread across {len(files)} PLOS ONE articles")

    n_journal = n_type = n_year = n_subfield = n_all = 0
    subfield_counts: Counter[str] = Counter()
    for f in sample:
        try:
            m = parse_metadata(f)
        except Exception:
            continue
        j = TARGET_JOURNAL_SUBSTRING in m.journal.lower()
        t = m.article_type == TARGET_ARTICLE_TYPE
        y = m.publication_date is not None and MIN_YEAR <= m.publication_date.year <= MAX_YEAR
        sf = m.psychology_subfields
        n_journal += j
        n_type += t
        n_year += y
        n_subfield += bool(sf)
        if j and t and y and sf:
            n_all += 1
            for s in sf:
                subfield_counts[s] += 1

    print(f"passed journal       : {n_journal}")
    print(f"passed article_type  : {n_type}")
    print(f"passed year {MIN_YEAR}-{MAX_YEAR} : {n_year}")
    print(f"passed psychology     : {n_subfield}")
    print(f"passed ALL           : {n_all}")
    if len(sample):
        print(f"  (~{100 * n_all / len(sample):.1f}% of sample; "
              f"extrapolates to ~{round(n_all / len(sample) * len(files), -2):.0f} articles)")
    print("--- psychology subfields captured (count in sample) ---")
    for name, count in subfield_counts.most_common():
        print(f"   {count:4d}  {name}")


if __name__ == "__main__":
    main()
