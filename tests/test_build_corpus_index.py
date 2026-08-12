import csv
import os

from src.build_corpus_index import build_index

FIXTURE_CORPUS_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "corpus")


def test_build_index_filters_to_only_matching_article(tmp_path):
    out_path = tmp_path / "index.csv"
    kept = build_index(FIXTURE_CORPUS_DIR, str(out_path))

    assert kept == 1
    with open(out_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row["doi"] == "10.1371/journal.pone.0012345"
    assert row["matched_subfields"] == "Social psychology"
    assert row["year"] == "2016"
    assert row["stage"] == "middle"
    assert row["lead_institution"] == "University of Virginia"


def test_build_index_excludes_animal_study(tmp_path):
    # animal_study.xml is a valid PLOS ONE psychology research article in range,
    # so it clears every other criterion — it must be dropped solely because its
    # subject taxonomy marks it as a non-human (animal) study.
    out_path = tmp_path / "index.csv"
    build_index(FIXTURE_CORPUS_DIR, str(out_path))
    with open(out_path, newline="", encoding="utf-8") as f:
        dois = {r["doi"] for r in csv.DictReader(f)}
    assert "10.1371/journal.pone.0099999" not in dois
