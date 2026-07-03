"""Investigate why psychology detection misses 2015 articles specifically.

For a corpus sample, buckets by year and counts, per year: total articles,
those our get_psychology_subfields() flags as psychology, and those whose raw
subject terms merely *contain* "psycholog". A year where the second count is
healthy but the first is ~0 is where our structural detection is failing.
Then dumps the raw <article-categories> of a few such failing 2015 articles
so the structural difference is visible.

    python scripts/diagnose_2015.py --corpus-dir "$PLOS_CORPUS"
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lxml import etree  # noqa: E402

from src.jats_xml import (  # noqa: E402
    get_publication_date,
    get_psychology_subfields,
    get_subjects,
    parse_tree,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", default=os.environ.get("PLOS_CORPUS", ""))
    parser.add_argument("--sample-size", type=int, default=6000)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.corpus_dir, "journal.pone.*.xml")))
    step = max(1, len(files) // args.sample_size)
    sample = files[::step]
    print(f"sampling {len(sample)} of {len(files)} PLOS ONE files")

    from collections import Counter

    per_year = defaultdict(lambda: {"n": 0, "psych_node": 0, "psych_substr": 0, "empty_subj": 0})
    dumps_2015 = []
    sgtype_2015: Counter = Counter()
    for f in sample:
        try:
            tree = parse_tree(f)
        except Exception:
            continue
        d = get_publication_date(tree)
        if d is None:
            continue
        subj, _ = get_subjects(tree)
        node_sf = get_psychology_subfields(tree)
        rec = per_year[d.year]
        rec["n"] += 1
        rec["psych_node"] += bool(node_sf)
        rec["psych_substr"] += any("psycholog" in s.lower() for s in subj)
        rec["empty_subj"] += not subj
        if d.year == 2015:
            cats = tree.find(".//article-categories")
            if cats is not None:
                for sg in cats.findall(".//subj-group"):
                    sgtype_2015[sg.get("subj-group-type")] += 1
            if len(dumps_2015) < 3:
                raw = etree.tostring(cats).decode()[:2000] if cats is not None else "<no article-categories>"
                dumps_2015.append((os.path.basename(f), raw))

    print("year | total | psych_by_node | psych_by_substring | empty_subject_list")
    for y in sorted(per_year):
        r = per_year[y]
        print(f"  {y} | {r['n']:4d} | {r['psych_node']:4d} | {r['psych_substr']:4d} | {r['empty_subj']:4d}")

    print("\n--- subj-group-type values seen in 2015 articles ---")
    for k, v in sgtype_2015.most_common():
        print(f"   {v:5d}  {k!r}")

    print("\n--- raw article-categories of first 3 sampled 2015 articles ---")
    for name, raw in dumps_2015:
        print(f"=== {name} ===")
        print(raw)
        print()


if __name__ == "__main__":
    main()
