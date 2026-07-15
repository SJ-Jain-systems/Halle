"""Make a blank CSV for hand-coding the pilot "gold" answers (see docs/SCORING_RUBRIC.md).

You read each pilot article yourself and fill in the same columns the model
spits out (the keys in src/extract_demographics.py, REQUIRED_KEYS), one row per
participant sample. Once it's filled in, src/score_pilot.py checks the model
against it.

Run it like:
    python -m src.make_gold_template \\
        --sample data/sampled_articles.csv \\
        --out data/pilot_gold_template.csv

We code per sample, not per article. If an article reports two or three separate
samples, give each one its own row (same doi, sample_id 1, 2, 3 and so on). The
template starts you off with one row per article at sample_id 1, so just copy the
row and bump sample_id when there's more than one sample.

The *_pct columns take a little JSON blob like {"male": 45, "female": 55}, the
same shape the model uses. Leave them as {} when nothing was reported.
ses_reported is the 0/1/2 scale (0 nothing, 1 just a category, 2 an actual
number); if there's a number, drop it in ses_value.

Pass --text-dir if you also want the methods text for each article dumped to a
file, so you're reading the exact same thing the model reads (via
src/jats_xml.get_extraction_text).
"""
from __future__ import annotations

import argparse
import csv
import os

from src.extract_demographics import REQUIRED_KEYS

# Columns we carry along so the coder has some context (these don't get scored),
# then the columns they actually fill in. doi doubles as a schema key.
HELPER_FIELDS = ["subfield", "title", "xml_path"]
SCHEMA_FIELDS = [
    "doi",
    "sample_id",
    "gender_reported",
    "gender_pct",
    "race_reported",
    "race_pct",
    "education_reported",
    "education_pct",
    "ses_reported",
    "ses_value",
]
TEMPLATE_FIELDS = ["doi"] + HELPER_FIELDS + [f for f in SCHEMA_FIELDS if f != "doi"]

# A blank row for the coder to fill in. The _pct cells start as {} so an
# untouched cell still parses.
_BLANK_CODING = {
    "sample_id": 1,
    "gender_reported": "",
    "gender_pct": "{}",
    "race_reported": "",
    "race_pct": "{}",
    "education_reported": "",
    "education_pct": "{}",
    "ses_reported": "",
    "ses_value": "",
}

assert set(SCHEMA_FIELDS) == REQUIRED_KEYS, (
    "SCHEMA_FIELDS must stay in sync with extract_demographics.REQUIRED_KEYS"
)


def load_sample(sample_csv: str) -> list[dict]:
    with open(sample_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_template_rows(articles: list[dict]) -> list[dict]:
    """One blank row per article, with the helper columns filled in."""
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
    """Dump each article's methods text so you read what the model reads. Returns
    how many files got written."""
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
        help="Also dump each article's extraction text here so you can read along.",
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
