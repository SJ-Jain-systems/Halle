import json
import os

import pytest

from src.extract_demographics import (
    ExtractionValidationError,
    REQUIRED_KEYS,
    extract_demographics_from_xml,
    format_field,
    parse_and_validate,
    parse_field,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_article.xml")

VALID_ROW = {
    "doi": "10.1371/journal.pone.0000001",
    "sample_id": 1,
    "gender": {"reported": 1, "pct": {"male": 45, "female": 55, "other": 0}},
    "race": {"reported": 0, "pct": {}},
    "education": {"reported": 0, "pct": {}},
    "ses": {"reported": 0, "value": None},
}


def test_parse_and_validate_accepts_well_formed_output():
    rows = parse_and_validate(json.dumps([VALID_ROW]), expected_doi=VALID_ROW["doi"])
    assert rows == [VALID_ROW]


def test_parse_and_validate_rejects_non_json():
    with pytest.raises(ExtractionValidationError):
        parse_and_validate("not json at all", expected_doi=VALID_ROW["doi"])


def test_parse_and_validate_rejects_missing_structural_key():
    bad_row = dict(VALID_ROW)
    del bad_row["sample_id"]
    with pytest.raises(ExtractionValidationError):
        parse_and_validate(json.dumps([bad_row]), expected_doi=VALID_ROW["doi"])


def test_parse_and_validate_defaults_missing_demographic():
    row = dict(VALID_ROW)
    del row["race"]
    out = parse_and_validate(json.dumps([row]), expected_doi=VALID_ROW["doi"])
    assert out[0]["race"] == {"reported": 0, "pct": {}}


def test_parse_and_validate_rejects_malformed_demographic_field():
    bad_row = dict(VALID_ROW)
    bad_row["gender"] = 1  # not an object with reported/pct
    with pytest.raises(ExtractionValidationError):
        parse_and_validate(json.dumps([bad_row]), expected_doi=VALID_ROW["doi"])


def test_parse_and_validate_rejects_doi_mismatch():
    with pytest.raises(ExtractionValidationError):
        parse_and_validate(json.dumps([VALID_ROW]), expected_doi="10.1371/journal.pone.9999999")


def test_required_keys_matches_valid_row_shape():
    assert REQUIRED_KEYS == set(VALID_ROW.keys())


def test_format_field_renders_flat_human_readable_form():
    assert format_field("gender", {"reported": 1, "pct": {"male": 60, "female": 40}}) == (
        "1, 60% Male, 40% Female"
    )
    assert format_field("race", {"reported": 0, "pct": {}}) == "0"
    assert format_field("ses", {"reported": 2, "value": 30000}) == "2, 30000"
    assert format_field("ses", {"reported": 0, "value": None}) == "0"


def test_parse_field_is_inverse_of_format_field():
    for name, value in [
        ("gender", {"reported": 1, "pct": {"male": 60, "female": 40}}),
        ("race", {"reported": 1, "pct": {"white": 70, "black": 30}}),
        ("education", {"reported": 0, "pct": {}}),
        ("ses", {"reported": 2, "value": "30000"}),
    ]:
        assert parse_field(name, format_field(name, value)) == value


class FakeClient:
    """Stands in for a real ModelClient so this test needs no GPU/model."""

    def __init__(self, response_rows):
        self.response_rows = response_rows
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return json.dumps(self.response_rows)


def test_extract_demographics_from_xml_reads_local_corpus_file():
    doi = "10.1371/journal.pone.0012345"
    row = dict(VALID_ROW)
    row["doi"] = doi
    client = FakeClient([row])

    rows = extract_demographics_from_xml(doi, FIXTURE, client)

    assert rows == [row]
    # The prompt should carry the article's actual text, not a placeholder.
    assert "45% male" in client.last_prompt
