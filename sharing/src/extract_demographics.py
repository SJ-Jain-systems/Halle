# NOTES
# This is where we tell the model what to do and check what it gives back. It
# holds the exact instruction we send with each paper, the list of fields every
# answer must have, and a strict checker that rejects anything malformed.
#
# The design that matters: one row per participant sample. If a paper ran three
# studies with three groups of people, we want three rows, all sharing the
# paper's ID. That's baked into the instruction.
#
# The checker is deliberately harsh. If the model returns broken data or invents
# an ID, we throw it out and flag it. Bad numbers here would corrupt the final
# trends, so we fail loudly instead of letting junk through.
"""The model instruction plus a strict checker for what it returns.

One row per participant sample. A paper with three samples gives three rows, all
sharing the same DOI.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from src.jats_xml import get_extraction_text

# Every answer row must contain exactly these fields. Anything missing, reject.
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

# The instruction sent to the model, with the paper's text pasted in at the
# bottom. It spells out the four demographics, the 0/1 "did they report it"
# flags, and demands clean JSON so the checker below can parse it.
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
    """Raised when the model's output doesn't match the required fields."""


@dataclass
class ModelClient:
    """Placeholder interface. The real runners in model_backend.py fill in
    generate(). Keeping this abstract means the extraction code doesn't care
    which model or hardware is used. Anything with a .generate() works.
    """

    model_id: str

    def generate(self, prompt: str) -> str:  # pragma: no cover - backend-specific
        raise NotImplementedError(
            "Wire this up to your inference backend (Hugging Face Inference API, "
            "a local transformers pipeline, etc.) before calling extract_demographics()."
        )


def build_prompt(doi: str, article_text: str) -> str:
    # Paste this paper's ID and text into the instruction template.
    return EXTRACTION_PROMPT_TEMPLATE.format(doi=doi, article_text=article_text)


def parse_and_validate(raw_output: str, expected_doi: str) -> list[dict]:
    """Turn the model's raw text into checked rows, or reject it.

    Three checks. Is it valid JSON. Is it a list. Does every row have all the
    required fields and the right paper ID. Any no, and we reject the whole
    thing. This is the wall that keeps bad data out of the study.
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
            # Guards against the model inventing a different paper's ID.
            raise ExtractionValidationError(
                f"Row {i} doi {row['doi']!r} does not match article doi {expected_doi!r}"
            )
    return rows


def extract_demographics(doi: str, article_text: str, client: ModelClient) -> list[dict]:
    # Straight line: build the prompt, ask the model, check the answer.
    prompt = build_prompt(doi, article_text)
    raw_output = client.generate(prompt)
    return parse_and_validate(raw_output, expected_doi=doi)


def extract_demographics_from_xml(doi: str, xml_path: str, client: ModelClient) -> list[dict]:
    """Read the paper's text off disk, then run the extraction.

    This is what the pipeline actually calls. No network. The file is already
    local.
    """
    article_text = get_extraction_text(xml_path)
    return extract_demographics(doi, article_text, client)
