"""Diagnose publication-date parsing across the corpus — investigating a
suspicious near-empty year (2015) in the built index. Samples PLOS ONE files,
reports the year histogram our parser produces, counts undated articles, and
dumps the raw <pub-date> markup so date-format changes across PLOS's history
are visible.

    python scripts/diagnose_dates.py --corpus-dir "$PLOS_CORPUS"
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lxml import etree  # noqa: E402

from src.jats_xml import get_publication_date, parse_tree  # noqa: E402


def raw_pubdates(tree) -> str:
    cats = tree.findall(".//pub-date")
    return " || ".join(etree.tostring(pd).decode().strip().replace("\n", "") for pd in cats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", default=os.environ.get("PLOS_CORPUS", ""))
    parser.add_argument("--sample-size", type=int, default=4000)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.corpus_dir, "journal.pone.*.xml")))
    step = max(1, len(files) // args.sample_size)
    sample = files[::step]
    print(f"sampling {len(sample)} of {len(files)} PLOS ONE files")

    years: Counter = Counter()
    undated = 0
    for f in sample:
        try:
            tree = parse_tree(f)
        except Exception:
            continue
        d = get_publication_date(tree)
        if d is None:
            undated += 1
            years["NONE"] += 1
            continue
        years[d.year] += 1

    print("year histogram (our parser):")
    for y in sorted(years, key=str):
        print(f"  {y}: {years[y]}")
    print(f"undated (publication_date is None): {undated}")

    # Dump raw pub-date markup for a few files that fall in the 2014-2016 window
    # by raw text, regardless of what our parser returns, to compare formats.
    print("\n--- raw <pub-date> markup samples (looking for 2014/2015/2016) ---")
    shown = {2014: 0, 2015: 0, 2016: 0}
    for f in sample:
        if all(v >= 2 for v in shown.values()):
            break
        try:
            tree = parse_tree(f)
        except Exception:
            continue
        raw = raw_pubdates(tree)
        for y in (2014, 2015, 2016):
            if f"<year>{y}</year>" in raw and shown[y] < 2:
                print(f"[{y}] {os.path.basename(f)}: {raw[:600]}")
                shown[y] += 1
                break


if __name__ == "__main__":
    main()
