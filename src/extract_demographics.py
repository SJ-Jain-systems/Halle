"""Turn full-text article XML into demographic-table rows (brief item "Extract
demographics" / item 2).

One row per distinct sample reported in an article — a single article with
three separate demographic samples yields three rows, all sharing the same
`doi`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

REQUIRED_KEYS = {
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
}

EXTRACTION_PROMPT_TEMPLATE = """\
You are coding a psychology research article for a systematic review of
demographic reporting. Read the article text below and, for EACH distinct
participant sample described (an article may report more than one), extract:

- gender: reported? (0/1), and percentage breakdown if reported (male/female/other)
- race: reported? (0/1), and percentage breakdown if reported (white/black/hispanic/asian/other)
- education (optional): reported? (0/1), percentage with college vs. no college if reported
- socioeconomic status (optional): 0 = not reported, 1 = reported as a category only
  (low/medium/high, no numbers), 2 = reported with a specific numeric threshold

Respond with ONLY a JSON array. Each element is one sample, with exactly these keys:
doi, sample_id, gender_reported, gender_pct, race_reported, race_pct,
education_reported, education_pct, ses_reported, ses_value.

Article DOI: {doi}

Article text:
{article_text}
"""


class ExtractionValidationError(ValueError):
    """Raised when a model's output doesn't match the required row schema."""


@dataclass
class ModelClient:
    """Minimal interface a model backend must implement."""

    model_id: str

    def generate(self, prompt: str) -> str:  # pragma: no cover - backend-specific
        raise NotImplementedError(
            "Wire this up to your inference backend (Hugging Face Inference API, "
            "a local transformers pipeline, etc.) before calling extract_demographics()."
        )


def build_prompt(doi: str, article_text: str) -> str:
    return EXTRACTION_PROMPT_TEMPLATE.format(doi=doi, article_text=article_text)


def parse_and_validate(raw_output: str, expected_doi: str) -> list[dict]:
    """Parse a model's raw text response into validated demographic rows.

    Raises ExtractionValidationError on malformed JSON or a missing/mismatched
    required field, rather than silently returning partial/garbage rows —
    correctness of this data feeds directly into the research analysis.
    """
    try:
        rows = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ExtractionValidationError(f"Model output was not valid JSON: {exc}") from exc

    if not isinstance(rows, list):
        raise ExtractionValidationError("Expected a JSON array of sample rows")

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ExtractionValidationError(f"Row {i} is not a JSON object")
        missing = REQUIRED_KEYS - row.keys()
        if missing:
            raise ExtractionValidationError(f"Row {i} missing required keys: {sorted(missing)}")
        if row["doi"] != expected_doi:
            raise ExtractionValidationError(
                f"Row {i} doi {row['doi']!r} does not match article doi {expected_doi!r}"
            )
    return rows


def extract_demographics(doi: str, article_text: str, client: ModelClient) -> list[dict]:
    prompt = build_prompt(doi, article_text)
    raw_output = client.generate(prompt)
    return parse_and_validate(raw_output, expected_doi=doi)
