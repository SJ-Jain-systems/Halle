import datetime as dt

from src.subfields import stage_for_date


def test_stage_for_date_buckets():
    assert stage_for_date(dt.date(2012, 1, 1)) == "early"
    assert stage_for_date(dt.date(2017, 6, 1)) == "middle"
    assert stage_for_date(dt.date(2021, 3, 1)) == "covid"
    assert stage_for_date(dt.date(2025, 12, 31)) == "post_covid"
    assert stage_for_date(dt.date(2009, 1, 1)) is None
