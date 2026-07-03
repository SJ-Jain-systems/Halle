"""Time-stage buckets for the analysis (brief item 1).

Psychology subfields are no longer a hardcoded list: the pipeline is
taxonomy-driven and captures *every* psychology subfield PLOS tags an article
with (see jats_xml.get_psychology_subfields), rather than matching against a
fixed set. The brief named five subfields, but PLOS's taxonomy doesn't use
one of them ("Quantitative psychology") verbatim and uses several others the
brief didn't list (Experimental psychology, Psychometrics, ...), so we take
whatever PLOS actually assigns and stratify/analyze over that.
"""
from __future__ import annotations

import datetime as _dt

# Time-stage buckets for the year-over-year / by-stage analysis (brief item 1).
TIME_STAGES = [
    ("early", 2010, 2014),
    ("middle", 2015, 2019),
    ("covid", 2020, 2023),
    ("post_covid", 2024, 2026),
]


def stage_for_date(pub_date: _dt.date, stages: list[tuple[str, int, int]] = TIME_STAGES) -> str | None:
    for name, start_year, end_year in stages:
        if start_year <= pub_date.year <= end_year:
            return name
    return None
