"""Generate a blank hand-coding template for the pilot ground-truth ("gold")
set (docs/SCORING_RUBRIC.md).

A human reads each pilot article and fills in, by hand, the same schema the
model emits (src/extract_demographics.py::REQUIRED_KEYS) — one row per distinct
participant sample. The completed file is the gold answer that
src/score_pilot.py scores the model's output against.

Usage:
    python -m src.make_gold_template \\
        --sample data/sampled_articles.csv \\
        --out data/pilot_gold_template.csv

The unit of analysis is the *sample*, not the article: an article that reports
several independent samples gets one row per sample, all sharing its `doi` and
numbered by `sample_id` (1, 2, ...). This template pre-fills one row per article
with `sample_id=1`; duplicate a row and bump `sample_id` for each extra sample.

Each demographic is one combined column in the flat, human-readable form the
pipeline uses (src/extract_demographics.format_field), e.g.
`gender` = `1, 45% Male, 55% Female`, `race` = `1, 60% White, 40% Black`
(0 = not reported, 1 = reported). Leave a cell blank or `0` when the
demographic isn't reported. `ses` uses the 0/1/2 detail scale (0 = not
reported, 1 = category only, 2 = numeric threshold) followed by the value if
any, e.g. `2, 30000` or `1, low`.

Optionally, --text-dir writes each article's methods-first extraction text
(exactly what the model reads, via src/jats_xml.get_extraction_text) to a file,
so the coder reads the same text the model does.
"""
from __future__ import annotations

import argparse
import csv
import os

from src.extract_demographics import REQUIRED_KEYS

# Helper columns carried through for the coder's convenience (not scored), then
# the schema columns they fill in. `doi` is both a helper and a schema key.
HELPER_FIELDS = ["subfield", "title", "xml_path"]
SCHEMA_FIELDS = [
    "doi",
    "sample_id",
    "gender",
    "race",
    "education",
    "ses",
]
TEMPLATE_FIELDS = ["doi"] + HELPER_FIELDS + [f for f in SCHEMA_FIELDS if f != "doi"]

# Blank template row: coder fills each combined demographic column in the flat
# form (e.g. "1, 45% Male, 55% Female"). Cells default to blank, which parses
# as "not reported".
_BLANK_CODING = {
    "sample_id": 1,
    "gender": "",
    "race": "",
    "education": "",
    "ses": "",
}

assert set(SCHEMA_FIELDS) == REQUIRED_KEYS, (
    "SCHEMA_FIELDS must stay in sync with extract_demographics.REQUIRED_KEYS"
)


def load_sample(sample_csv: str) -> list[dict]:
    with open(sample_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_template_rows(articles: list[dict]) -> list[dict]:
    """One blank coding row per article, carrying the helper columns."""
    rows = []
    for a in articles:
        row = {"doi": a.get("doi", "")}
        for h in HELPER_FIELDS:
            row[h] = a.get(h, "")
        row.update(_BLANK_CODING)
        rows.append(row)
    return rows


def write_template(rows: list[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TEMPLATE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def dump_texts(articles: list[dict], text_dir: str) -> int:
    """Write each article's methods-first extraction text so the coder reads the
    same text the model sees. Returns the number of files written."""
    from src.jats_xml import get_extraction_text

    os.makedirs(text_dir, exist_ok=True)
    written = 0
    for a in articles:
        xml_path = a.get("xml_path", "")
        if not xml_path or not os.path.exists(xml_path):
            continue
        text = get_extraction_text(xml_path)
        out = os.path.join(text_dir, f"{a['doi'].replace('/', '_')}.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", default="data/sampled_articles.csv")
    parser.add_argument("--out", default="data/pilot_gold_template.csv")
    parser.add_argument(
        "--text-dir",
        default=None,
        help="Optional: also write each article's extraction text here for the coder.",
    )
    args = parser.parse_args()

    articles = load_sample(args.sample)
    rows = build_template_rows(articles)
    write_template(rows, args.out)
    print(f"Wrote {len(rows)} blank coding rows to {args.out}")
    if args.text_dir:
        n = dump_texts(articles, args.text_dir)
        print(f"Wrote extraction text for {n} articles to {args.text_dir}")


if __name__ == "__main__":
    main()
