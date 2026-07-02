"""Psychology subfields in scope for this study (brief item 3.c)."""
from __future__ import annotations

import datetime as _dt

PSYCHOLOGY_SUBFIELDS = [
    "Social psychology",
    "Cognitive psychology",
    "Developmental psychology",
    "Clinical psychology",
    "Quantitative psychology",
]

# Time-stage buckets for the year-over-year / by-stage analysis (brief item 1).
TIME_STAGES = [
    ("early", 2010, 2014),
    ("middle", 2015, 2019),
    ("covid", 2020, 2023),
    ("post_covid", 2024, 2026),
]


def matched_subfields(subject_terms: list[str], subfields: list[str] = PSYCHOLOGY_SUBFIELDS) -> list[str]:
    """Which of our target subfields a PLOS taxonomy term list touches.

    Matches on the full `subject` term list (see src/jats_xml.py::get_subjects),
    not just `subject_level_1` — PLOS's taxonomy nests Psychology under a
    top-level Discipline like "Biology and life sciences" or "Social
    sciences", so subfield names like "Social psychology" show up several
    levels deep, never as a level-1 term.
    """
    lowered_terms = [t.lower() for t in subject_terms]
    return [
        subfield
        for subfield in subfields
        if any(subfield.lower() in term for term in lowered_terms)
    ]


def stage_for_date(pub_date: _dt.date, stages: list[tuple[str, int, int]] = TIME_STAGES) -> str | None:
    for name, start_year, end_year in stages:
        if start_year <= pub_date.year <= end_year:
            return name
    return None
