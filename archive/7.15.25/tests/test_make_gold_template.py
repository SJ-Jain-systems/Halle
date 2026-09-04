import csv

from src.extract_demographics import REQUIRED_KEYS
from src.make_gold_template import (
    SCHEMA_FIELDS,
    TEMPLATE_FIELDS,
    build_template_rows,
    write_template,
)

ARTICLES = [
    {"doi": "10.1371/journal.pone.0000001", "subfield": "Social psychology", "title": "A", "xml_path": "/a.xml"},
    {"doi": "10.1371/journal.pone.0000002", "subfield": "Cognitive psychology", "title": "B", "xml_path": "/b.xml"},
]


def test_schema_fields_match_extraction_required_keys():
    assert set(SCHEMA_FIELDS) == REQUIRED_KEYS


def test_build_template_rows_one_row_per_article_with_helpers():
    rows = build_template_rows(ARTICLES)
    assert len(rows) == len(ARTICLES)
    first = rows[0]
    # Helper columns carried through, coding columns blank-but-present.
    assert first["doi"] == ARTICLES[0]["doi"]
    assert first["subfield"] == "Social psychology"
    assert first["sample_id"] == 1
    assert first["gender_reported"] == ""
    assert first["gender_pct"] == "{}"


def test_write_template_header_and_contents(tmp_path):
    out = tmp_path / "pilot_gold_template.csv"
    write_template(build_template_rows(ARTICLES), str(out))
    with open(out, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == TEMPLATE_FIELDS
        loaded = list(reader)
    assert [r["doi"] for r in loaded] == [a["doi"] for a in ARTICLES]
    # Every required schema column is writable in the template.
    for key in REQUIRED_KEYS:
        assert key in reader.fieldnames
