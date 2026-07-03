# ============================================================================
# PLAIN-ENGLISH NOTES (for colleagues reading this file)
#
# What this file is for: this is where we tell the model what to do and check
# what it gives back. It holds:
#   - the exact instruction (the "prompt") we send with each paper,
#   - the list of fields every answer must contain,
#   - a strict checker that rejects anything malformed.
#
# The design that matters: ONE ROW PER PARTICIPANT SAMPLE. If a paper ran three
# studies with three different groups of people, we want three rows, all sharing
# the paper's ID. That is baked into the instruction.
#
# The checker (parse_and_validate) is deliberately harsh. If the model returns
# broken data or invents an ID, we throw it out and flag it rather than let bad
# numbers quietly pollute the dataset. Garbage in the demographics table would
# corrupt the final trends, so we would rather fail loudly.
# ============================================================================

"""Turn full-text article XML into demographic-table rows (brief item "Extract
demographics" / item 2).

One row per distinct sample reported in an article — a single article with
three separate demographic samples yields three rows, all sharing the same
`doi`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from src.jats_xml import get_extraction_text

# Every answer row MUST contain exactly these fields. Anything missing = reject.
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

# The actual instruction sent to the model, with the paper's text pasted in at
# the bottom. It spells out the four demographics, the 0/1 "did they report it"
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
    """Raised when a model's output doesn't match the required row schema."""


@dataclass
class ModelClient:
    """Minimal interface a model backend must implement."""

    # This is a placeholder. The real model runners (in model_backend.py) fill in
    # generate(). Keeping this abstract means the extraction logic doesn't care
    # WHICH model or hardware is used - anything with a .generate() works.
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
    """Parse a model's raw text response into validated demographic rows.

    Raises ExtractionValidationError on malformed JSON or a missing/mismatched
    required field, rather than silently returning partial/garbage rows —
    correctness of this data feeds directly into the research analysis.
    """
    # Step 1: is it even valid JSON? Step 2: is it a list? Step 3: does every row
    # have all the required fields and the right paper ID? Any "no" and we reject
    # the whole thing. This is the wall that keeps bad data out of the study.
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
            # Guards against the model hallucinating a different paper's ID.
            raise ExtractionValidationError(
                f"Row {i} doi {row['doi']!r} does not match article doi {expected_doi!r}"
            )
    return rows


def extract_demographics(doi: str, article_text: str, client: ModelClient) -> list[dict]:
    # The straight-line version: build the prompt, ask the model, check the answer.
    prompt = build_prompt(doi, article_text)
    raw_output = client.generate(prompt)
    return parse_and_validate(raw_output, expected_doi=doi)


def extract_demographics_from_xml(doi: str, xml_path: str, client: ModelClient) -> list[dict]:
    """Convenience wrapper: pull Methods/Participants-first text straight out
    of the local corpus XML (src/jats_xml.py) and extract from it. This is
    what src/run_pipeline.py and src/run_pilot.py actually call — nothing
    downloads full text over the network, it's already on disk.
    """
    # The version the pipeline actually calls: read the paper's text off disk,
    # then run the extraction. No network involved - the file is already local.
    article_text = get_extraction_text(xml_path)
    return extract_demographics(doi, article_text, client)
