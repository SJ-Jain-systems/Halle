"""Turn full-text article XML into demographic-table rows (brief item "Extract
demographics" / item 2).

One row per distinct sample reported in an article — a single article with
three separate demographic samples yields three rows, all sharing the same
`doi`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.jats_xml import get_extraction_text

# One combined field per demographic (rather than a separate reported-flag and
# percentage column each): gender/race/education carry {"reported": 0/1, "pct":
# {...}} and ses carries {"reported": 0/1/2, "value": ...}. This is what the
# model emits and what parse_and_validate() checks.
REQUIRED_KEYS = {
    "doi",
    "sample_id",
    "gender",
    "race",
    "education",
    "ses",
}

# Canonical subgroup order + display labels for the percentage demographics.
# Order here is the order subgroups appear in the flat, human-readable form
# (see format_field): "1, 60% White, 20% Black, ...".
PCT_SUBGROUPS = {
    "gender": [("male", "Male"), ("female", "Female"), ("other", "Other")],
    "race": [
        ("white", "White"),
        ("black", "Black"),
        ("hispanic", "Hispanic"),
        ("asian", "Asian"),
        ("other", "Other"),
    ],
    "education": [("college", "College"), ("no_college", "No college")],
}
PCT_FIELDS = ("gender", "race", "education")
DEMOGRAPHIC_FIELDS = ("gender", "race", "education", "ses")

EXTRACTION_PROMPT_TEMPLATE = """\
You are coding a psychology research article for a systematic review of
demographic reporting. Read the article text below and, for EACH distinct
participant sample described (an article may report more than one), extract:

- gender: reported? (0/1), and percentage breakdown if reported (male/female/other)
- race: reported? (0/1), and percentage breakdown if reported (white/black/hispanic/asian/other)
- education (optional): reported? (0/1), percentage with college vs. no college if reported
- socioeconomic status (optional): 0 = not reported, 1 = reported as a category only
  (low/medium/high, no numbers), 2 = reported with a specific numeric threshold

Respond with ONLY a JSON array. Each element is one sample. Combine each
demographic into a single field, with EXACTLY these keys:

  "doi", "sample_id",
  "gender":    {{"reported": 0 or 1, "pct": {{"male": .., "female": .., "other": ..}}}},
  "race":      {{"reported": 0 or 1, "pct": {{"white": .., "black": .., "hispanic": .., "asian": .., "other": ..}}}},
  "education": {{"reported": 0 or 1, "pct": {{"college": .., "no_college": ..}}}},
  "ses":       {{"reported": 0, 1, or 2, "value": <numeric threshold, category label, or null>}}

`reported` is 1 when the demographic is reported for that sample and 0 when it
isn't (for ses: 0 = not reported, 1 = category only such as low/medium/high,
2 = reported with a specific numeric threshold). Set `pct` to {{}} when
`reported` is 0.

Output ONLY the JSON array. No explanation, no markdown fences, no commentary,
and nothing after the closing ]. Do not repeat text. Your entire response must
start with [ and end with ]. Example of the exact format (one sample, values
illustrative):

[{{"doi": "{doi}", "sample_id": 1, "gender": {{"reported": 1, "pct": {{"male": 40, "female": 60}}}}, "race": {{"reported": 0, "pct": {{}}}}, "education": {{"reported": 0, "pct": {{}}}}, "ses": {{"reported": 0, "value": null}}}}]

Article DOI: {doi}

Article text:
{article_text}
"""

_PCT_TOKEN = re.compile(r"^\s*(-?[\d.]+)\s*%\s*(.+?)\s*$")


def _fmt_num(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number.is_integer() else str(number)


def format_field(name: str, value) -> str:
    """Serialize a combined demographic field to the flat, human-readable form
    used in the output table and the gold sheet:

        gender {"reported": 1, "pct": {"male": 60, "female": 40}}
            -> "1, 60% Male, 40% Female"
        race not reported ({"reported": 0, ...})   -> "0"
        ses {"reported": 2, "value": 30000}        -> "2, 30000"

    (0 = not reported, 1 = reported.)
    """
    value = value or {}
    reported = int(value.get("reported") or 0)
    if name == "ses":
        raw = value.get("value")
        if raw in (None, ""):
            return str(reported)
        return f"{reported}, {raw}"
    if not reported:
        return "0"
    pct = value.get("pct") or {}
    labels = dict(PCT_SUBGROUPS[name])
    parts = ["1"]
    for key, label in PCT_SUBGROUPS[name]:
        if key in pct:
            parts.append(f"{_fmt_num(pct[key])}% {label}")
    # Preserve any non-canonical subgroup the model reported, rather than drop it.
    for key, subval in pct.items():
        if key not in labels:
            parts.append(f"{_fmt_num(subval)}% {key.replace('_', ' ').title()}")
    return ", ".join(parts)


def parse_field(name: str, text) -> dict:
    """Inverse of format_field: a flat string back into the combined dict.
    Passes a dict straight through, so callers can hand it either form."""
    if isinstance(text, dict):
        return text
    s = (text or "").strip()
    if name == "ses":
        if s == "":
            return {"reported": 0, "value": None}
        head, _, tail = s.partition(",")
        reported = int(float(head.strip())) if head.strip() else 0
        value = tail.strip() or None
        return {"reported": reported, "value": value}
    if s == "" or s == "0":
        return {"reported": 0, "pct": {}}
    parts = [p.strip() for p in s.split(",")]
    reported = int(float(parts[0])) if parts[0] else 0
    label_to_key = {label.lower(): key for key, label in PCT_SUBGROUPS[name]}
    pct: dict = {}
    for token in parts[1:]:
        m = _PCT_TOKEN.match(token)
        if not m:
            continue
        number = float(m.group(1))
        number = int(number) if number.is_integer() else number
        label = m.group(2).strip().lower()
        key = label_to_key.get(label, label.replace(" ", "_"))
        pct[key] = number
    return {"reported": reported, "pct": pct}


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


def _extract_json_array(raw: str) -> str:
    """Best-effort pull of the JSON array out of a model response that may wrap
    it in markdown fences or surrounding prose (instruct models often do, even
    when told to emit only JSON). Strips a leading/trailing ``` fence, then
    returns the first balanced top-level ``[...]`` span, ignoring brackets
    inside strings. Falls back to the stripped text so json.loads still raises a
    clear error when there's no array to find.
    """
    s = (raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[A-Za-z0-9_-]*[ \t]*\r?\n", "", s)
        s = re.sub(r"\r?\n```[ \t]*$", "", s).strip()
    start = s.find("[")
    if start == -1:
        return s
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return s[start:]


def parse_and_validate(raw_output: str, expected_doi: str) -> list[dict]:
    """Parse a model's raw text response into validated demographic rows.

    Raises ExtractionValidationError on malformed JSON or a missing/mismatched
    required field, rather than silently returning partial/garbage rows —
    correctness of this data feeds directly into the research analysis. The
    offending raw text is attached to the exception as ``raw_output`` so callers
    (src/run_pilot.py) can record it for debugging.
    """
    try:
        return _parse_and_validate(raw_output, expected_doi)
    except ExtractionValidationError as exc:
        if not hasattr(exc, "raw_output"):
            exc.raw_output = raw_output
        raise


def _repair_truncated_array(payload: str) -> list | None:
    """Salvage a JSON array truncated mid-object (the model hit the token cap
    before closing the array): drop the trailing incomplete element and close
    at the last complete object. Returns the parsed list, or None if nothing
    recoverable."""
    s = payload.strip()
    if not s.startswith("["):
        return None
    for m in reversed([mm.start() for mm in re.finditer(r"\}", s)][-80:]):
        candidate = s[: m + 1].rstrip().rstrip(",").rstrip() + "]"
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, RecursionError):
            continue
        if isinstance(value, list) and value:
            return value
    return None


def _parse_and_validate(raw_output: str, expected_doi: str) -> list[dict]:
    payload = _extract_json_array(raw_output)
    try:
        rows = json.loads(payload)
    except (json.JSONDecodeError, RecursionError) as exc:
        # The model wrapped, rambled past, or (at the token cap) truncated its
        # array. Try to salvage the complete leading objects before giving up.
        # RecursionError: a degenerate generation (runaway nested brackets) that
        # json's recursive decoder can't handle — treat it as malformed output.
        rows = _repair_truncated_array(payload)
        if rows is None:
            raise ExtractionValidationError(f"Model output was not valid JSON: {exc}") from exc

    if not isinstance(rows, list):
        raise ExtractionValidationError("Expected a JSON array of sample rows")

    rows = [r for r in rows if isinstance(r, dict)]
    if not rows:
        raise ExtractionValidationError("No JSON object rows in model output")

    for i, row in enumerate(rows):
        # The article's DOI is known from the pipeline; don't require the model
        # to echo it back correctly. Default a missing sample_id to position.
        row["doi"] = expected_doi
        row.setdefault("sample_id", i + 1)
        # Coerce each demographic into the canonical shape rather than discard
        # the row: the model often shortcuts a not-reported field to a bare 0
        # (or 1) instead of {"reported": .., "pct": {}}. A gold-reported
        # demographic the model dropped is still counted against recall, so this
        # salvages the row's real data without hiding model errors.
        for name in PCT_FIELDS:
            row[name] = _coerce_pct_field(row.get(name))
        row["ses"] = _coerce_ses_field(row.get("ses"))
    return rows


def _coerce_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_pct_field(field) -> dict:
    """Normalize a gender/race/education field to {"reported": int, "pct": dict}."""
    if isinstance(field, bool):
        return {"reported": int(field), "pct": {}}
    if isinstance(field, (int, float)):
        return {"reported": int(field), "pct": {}}
    if isinstance(field, dict):
        pct = field.get("pct")
        if not isinstance(pct, dict):
            pct = {}
        # If the model gave percentages but omitted the flag, treat as reported.
        reported = _coerce_int(field.get("reported", 1 if pct else 0))
        return {"reported": reported, "pct": pct}
    return {"reported": 0, "pct": {}}


def _coerce_ses_field(field) -> dict:
    """Normalize the ses field to {"reported": int, "value": <any|None>}."""
    if isinstance(field, bool):
        return {"reported": int(field), "value": None}
    if isinstance(field, (int, float)):
        return {"reported": int(field), "value": None}
    if isinstance(field, dict):
        return {"reported": _coerce_int(field.get("reported", 0)), "value": field.get("value")}
    return {"reported": 0, "value": None}


def extract_demographics(doi: str, article_text: str, client: ModelClient) -> list[dict]:
    prompt = build_prompt(doi, article_text)
    raw_output = client.generate(prompt)
    return parse_and_validate(raw_output, expected_doi=doi)


def extract_demographics_from_xml(doi: str, xml_path: str, client: ModelClient) -> list[dict]:
    """Convenience wrapper: pull Methods/Participants-first text straight out
    of the local corpus XML (src/jats_xml.py) and extract from it. This is
    what src/run_pipeline.py and src/run_pilot.py actually call — nothing
    downloads full text over the network, it's already on disk.
    """
    article_text = get_extraction_text(xml_path)
    return extract_demographics(doi, article_text, client)
