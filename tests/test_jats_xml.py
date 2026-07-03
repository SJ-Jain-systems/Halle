import datetime as dt
import os

from src.jats_xml import (
    get_extraction_text,
    get_psychology_subfields,
    get_subjects,
    parse_metadata,
    parse_tree,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_article.xml")

# Real-corpus taxonomy structure: PLOS's current thesaurus tags discipline
# groups `Discipline-v3` (not `Discipline`), with a `heading` group for the
# article type and deeply nested inner subj-groups carrying no type at all.
# Verified against journal.pone.0320306.xml in the allofplos corpus.
DISCIPLINE_V3_XML = """<?xml version="1.0"?>
<article article-type="research-article">
  <front><journal-meta><journal-title>PLOS ONE</journal-title></journal-meta>
  <article-meta>
    <article-id pub-id-type="doi">10.1371/journal.pone.0320306</article-id>
    <article-categories>
      <subj-group subj-group-type="heading"><subject>Research Article</subject></subj-group>
      <subj-group subj-group-type="Discipline-v3">
        <subject>Biology and life sciences</subject>
        <subj-group><subject>Psychology</subject>
          <subj-group><subject>Social psychology</subject></subj-group></subj-group>
      </subj-group>
      <subj-group subj-group-type="Discipline-v3">
        <subject>Social sciences</subject>
        <subj-group><subject>Psychology</subject>
          <subj-group><subject>Social psychology</subject></subj-group></subj-group>
      </subj-group>
    </article-categories>
  </article-meta></front>
</article>
"""


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


def test_get_subjects_reads_discipline_v3(tmp_path):
    path = tmp_path / "v3.xml"
    path.write_text(DISCIPLINE_V3_XML, encoding="utf-8")
    subject, subject_level_1 = get_subjects(parse_tree(str(path)))
    # `heading` (Research Article) must be skipped; both Discipline-v3 branches read.
    assert subject_level_1 == ["Biology and life sciences", "Social sciences"]
    assert "Social psychology" in subject
    assert "Psychology" in subject
    assert "Research Article" not in subject


def test_get_psychology_subfields_v3(tmp_path):
    path = tmp_path / "v3.xml"
    path.write_text(DISCIPLINE_V3_XML, encoding="utf-8")
    # "Social psychology" is nested under Psychology in two discipline branches;
    # it should come back once, deduplicated.
    assert get_psychology_subfields(parse_tree(str(path))) == ["Social psychology"]


def test_get_psychology_subfields_fixture():
    # The old-style `Discipline` fixture also nests Social psychology under Psychology.
    assert get_psychology_subfields(parse_tree(FIXTURE)) == ["Social psychology"]


def test_get_psychology_subfields_empty_when_no_psychology(tmp_path):
    xml = """<?xml version="1.0"?>
<article article-type="research-article"><front><article-meta>
<article-categories>
  <subj-group subj-group-type="Discipline-v3"><subject>Medicine and health sciences</subject>
    <subj-group><subject>Oncology</subject></subj-group></subj-group>
</article-categories>
</article-meta></front></article>"""
    path = tmp_path / "no_psych.xml"
    path.write_text(xml, encoding="utf-8")
    assert get_psychology_subfields(parse_tree(str(path))) == []
