import datetime as dt

from src.subfields import matched_subfields, stage_for_date


def test_matched_subfields_finds_nested_term():
    terms = ["Biology and life sciences", "Psychology", "Social psychology"]
    assert matched_subfields(terms) == ["Social psychology"]


def test_matched_subfields_no_match():
    terms = ["Medicine and health sciences", "Oncology"]
    assert matched_subfields(terms) == []


def test_matched_subfields_multiple():
    terms = ["Cognitive psychology", "Clinical psychology"]
    result = matched_subfields(terms)
    assert set(result) == {"Cognitive psychology", "Clinical psychology"}


def test_stage_for_date_buckets():
    assert stage_for_date(dt.date(2012, 1, 1)) == "early"
    assert stage_for_date(dt.date(2017, 6, 1)) == "middle"
    assert stage_for_date(dt.date(2021, 3, 1)) == "covid"
    assert stage_for_date(dt.date(2025, 12, 31)) == "post_covid"
    assert stage_for_date(dt.date(2009, 1, 1)) is None
