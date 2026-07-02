import datetime as dt
import os

from src.jats_xml import get_extraction_text, parse_metadata

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_article.xml")


def test_parse_metadata_basic_fields():
    meta = parse_metadata(FIXTURE)
    assert meta.doi == "10.1371/journal.pone.0012345"
    assert meta.title == "A study of social attitudes"
    assert meta.journal == "PLOS ONE"
    assert meta.article_type == "research-article"
    assert meta.publication_date == dt.date(2016, 5, 10)


def test_parse_metadata_subjects_top_level_vs_all_terms():
    meta = parse_metadata(FIXTURE)
    assert meta.subject_level_1 == ["Biology and life sciences", "Medicine and health sciences"]
    assert "Social psychology" in meta.subject
    assert "Psychology" in meta.subject
    assert "Biology and life sciences" in meta.subject


def test_parse_metadata_lead_institution_prefers_corresponding_author():
    meta = parse_metadata(FIXTURE)
    assert meta.lead_institution == "University of Virginia"


def test_get_extraction_text_prioritizes_methods_section():
    text = get_extraction_text(FIXTURE)
    methods_pos = text.find("final sample included 200 participants")
    intro_pos = text.find("examines social attitudes")
    assert methods_pos != -1
    assert intro_pos != -1
    assert methods_pos < intro_pos


def test_get_extraction_text_does_not_duplicate_nested_sections():
    text = get_extraction_text(FIXTURE)
    assert text.count("45% male") == 1
