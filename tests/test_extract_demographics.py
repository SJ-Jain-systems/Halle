import json
import os

import pytest

from src.extract_demographics import (
    ExtractionValidationError,
    REQUIRED_KEYS,
    extract_demographics_from_xml,
    parse_and_validate,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_article.xml")

VALID_ROW = {
    "doi": "10.1371/journal.pone.0000001",
    "sample_id": 1,
    "gender_reported": 1,
    "gender_pct": {"male": 45, "female": 55, "other": 0},
    "race_reported": 0,
    "race_pct": {},
    "education_reported": 0,
    "education_pct": {},
    "ses_reported": 0,
    "ses_value": None,
}


def test_parse_and_validate_accepts_well_formed_output():
    rows = parse_and_validate(json.dumps([VALID_ROW]), expected_doi=VALID_ROW["doi"])
    assert rows == [VALID_ROW]


def test_parse_and_validate_rejects_non_json():
    with pytest.raises(ExtractionValidationError):
        parse_and_validate("not json at all", expected_doi=VALID_ROW["doi"])


def test_parse_and_validate_rejects_missing_keys():
    bad_row = dict(VALID_ROW)
    del bad_row["race_pct"]
    with pytest.raises(ExtractionValidationError):
        parse_and_validate(json.dumps([bad_row]), expected_doi=VALID_ROW["doi"])


def test_parse_and_validate_rejects_doi_mismatch():
    with pytest.raises(ExtractionValidationError):
        parse_and_validate(json.dumps([VALID_ROW]), expected_doi="10.1371/journal.pone.9999999")


def test_required_keys_matches_valid_row_shape():
    assert REQUIRED_KEYS == set(VALID_ROW.keys())


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
