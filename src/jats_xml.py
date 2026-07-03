"""Parse JATS XML (the format used by every article in the allofplos corpus)
directly with lxml, rather than relying on allofplos's higher-level Article
wrapper. The wrapper doesn't expose subject taxonomy or a `subject_level_1`
equivalent, and that's exactly what brief item 4.c.i needs (previously "via
Solr" — now derived locally from the XML itself, since we're no longer
calling Solr at all).

JATS structure assumed here (standard PLOS layout — spot-check a handful of
articles against this after the corpus is downloaded, since taxonomy nesting
has changed slightly across PLOS's history):

  <article article-type="research-article">
    <front>
      <journal-meta>...<journal-title>PLOS ONE</journal-title>...</journal-meta>
      <article-meta>
        <pub-date pub-type="epub"><year/><month/><day/></pub-date>
        <contrib-group>
          <contrib contrib-type="author" corresp="yes">
            <xref ref-type="aff" rid="aff1"/>
          </contrib>
        </contrib-group>
        <aff id="aff1"><institution>...</institution></aff>
        <article-categories>
          <subj-group subj-group-type="Discipline">
            <subject>Biology and life sciences</subject>
            <subj-group>
              <subject>Psychology</subject>
              <subj-group>
                <subject>Social psychology</subject>
              </subj-group>
            </subj-group>
          </subj-group>
        </article-categories>
      </article-meta>
    </front>
    <body>
      <sec><title>Methods</title>...<sec><title>Participants</title>...</sec></sec>
    </body>
  </article>
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

from lxml import etree

SECTION_TITLE_KEYWORDS = ("method", "participant", "sample", "procedure")


@dataclass
class ArticleMetadata:
    doi: str
    title: str
    journal: str
    article_type: str
    publication_date: _dt.date | None
    subject_level_1: list[str] = field(default_factory=list)
    subject: list[str] = field(default_factory=list)
    psychology_subfields: list[str] = field(default_factory=list)
    lead_institution: str | None = None
    xml_path: str = ""


def parse_tree(xml_path: str) -> etree._ElementTree:
    return etree.parse(xml_path)


def get_doi(tree: etree._ElementTree) -> str:
    node = tree.find(".//article-id[@pub-id-type='doi']")
    return node.text.strip() if node is not None and node.text else ""


def get_title(tree: etree._ElementTree) -> str:
    node = tree.find(".//article-title")
    return "".join(node.itertext()).strip() if node is not None else ""


def get_journal(tree: etree._ElementTree) -> str:
    node = tree.find(".//journal-meta//journal-title")
    return node.text.strip() if node is not None and node.text else ""


def get_article_type(tree: etree._ElementTree) -> str:
    root = tree.getroot()
    return root.get("article-type", "")


def get_publication_date(tree: etree._ElementTree) -> _dt.date | None:
    for pub_type in ("epub", "collection", "ppub", None):
        xpath = ".//pub-date[@pub-type='%s']" % pub_type if pub_type else ".//pub-date"
        node = tree.find(xpath)
        if node is None:
            continue
        year = node.findtext("year")
        month = node.findtext("month") or "1"
        day = node.findtext("day") or "1"
        if not year:
            continue
        try:
            return _dt.date(int(year), int(month), int(day))
        except ValueError:
            continue
    return None


def get_subjects(tree: etree._ElementTree) -> tuple[list[str], list[str]]:
    """Returns (subject, subject_level_1), replicating the Solr fields of the
    same name: `subject` is every taxonomy term tagged on the article, in
    document order and deduplicated; `subject_level_1` is just the top-level
    term of each Discipline branch (an article can carry more than one
    Discipline, e.g. Biology and Social sciences both).

    Matches any `subj-group-type` that starts with "Discipline" — verified
    against the real allofplos corpus, PLOS's current thesaurus tags these
    groups `Discipline-v3`, while older articles use plain `Discipline`. The
    other group types present (`heading` for "Research Article", nested inner
    groups with no type) are correctly skipped.
    """
    subject: list[str] = []
    subject_level_1: list[str] = []
    cats = tree.find(".//article-categories")
    if cats is None:
        return subject, subject_level_1
    for group in cats.findall("subj-group"):
        group_type = group.get("subj-group-type") or ""
        if not group_type.startswith("Discipline"):
            continue
        top_subject = group.find("subject")
        if top_subject is not None and top_subject.text:
            top_text = top_subject.text.strip()
            if top_text not in subject_level_1:
                subject_level_1.append(top_text)
        for subject_node in group.findall(".//subject"):
            if subject_node.text:
                text = subject_node.text.strip()
                if text not in subject:
                    subject.append(text)
    return subject, subject_level_1


def get_psychology_subfields(tree: etree._ElementTree) -> list[str]:
    """Every taxonomy term nested directly under a "Psychology" node, across
    all Discipline groups — i.e. the article's psychology subfield(s).

    For `Biology and life sciences > Psychology > Social psychology` this
    returns `["Social psychology"]`. An article tagged under Psychology in
    more than one discipline branch (PLOS often files psychology under both
    "Biology and life sciences" and "Social sciences") yields the subfield
    once, deduplicated. If "Psychology" is tagged as a leaf with no subfield
    child, returns `["Psychology"]` so the article still counts as psychology
    with an unspecified subfield. Empty if the article isn't under Psychology
    at all.

    This is taxonomy-driven rather than matched against a fixed subfield list,
    so it captures *all* psychology subfields PLOS uses (Social, Cognitive,
    Clinical, Developmental, Experimental psychology, Psychometrics, ...),
    not just a hardcoded few.
    """
    subfields: list[str] = []
    cats = tree.find(".//article-categories")
    if cats is None:
        return subfields
    for subject_node in cats.iter("subject"):
        if (subject_node.text or "").strip() != "Psychology":
            continue
        parent_group = subject_node.getparent()
        if parent_group is None:
            continue
        child_groups = [c for c in parent_group if c.tag == "subj-group"]
        if not child_groups:
            if "Psychology" not in subfields:
                subfields.append("Psychology")
            continue
        for child_group in child_groups:
            child_subject = child_group.find("subject")
            if child_subject is not None and child_subject.text:
                term = child_subject.text.strip()
                if term not in subfields:
                    subfields.append(term)
    return subfields


def get_lead_institution(tree: etree._ElementTree) -> str | None:
    """Corresponding author's institution; falls back to the first listed
    author if no contrib is marked corresp="yes" (brief item 4.b: "Corresponding
    (First) Author institution")."""
    contribs = tree.findall(".//article-meta/contrib-group/contrib[@contrib-type='author']")
    if not contribs:
        return None
    lead = next((c for c in contribs if c.get("corresp") == "yes"), contribs[0])
    xref = lead.find("xref[@ref-type='aff']")
    if xref is None:
        return None
    rid = xref.get("rid")
    if not rid:
        return None
    aff = tree.find(f".//aff[@id='{rid}']")
    if aff is None:
        return None
    institution = aff.find("institution")
    if institution is not None and institution.text:
        return institution.text.strip()
    # No structured <institution>; fall back to the affiliation's full text.
    text = "".join(aff.itertext()).strip()
    return text or None


def parse_metadata(xml_path: str) -> ArticleMetadata:
    tree = parse_tree(xml_path)
    subject, subject_level_1 = get_subjects(tree)
    return ArticleMetadata(
        doi=get_doi(tree),
        title=get_title(tree),
        journal=get_journal(tree),
        article_type=get_article_type(tree),
        publication_date=get_publication_date(tree),
        subject_level_1=subject_level_1,
        subject=subject,
        psychology_subfields=get_psychology_subfields(tree),
        lead_institution=get_lead_institution(tree),
        xml_path=xml_path,
    )


def get_extraction_text(xml_path: str) -> str:
    """Full body text, but with Methods/Participants/Sample/Procedure
    sections moved to the front — demographics almost always live there, and
    putting them first keeps the signal near the start of the prompt for
    articles long enough to worry about context budget.
    """
    tree = parse_tree(xml_path)
    body = tree.find(".//body")
    if body is None:
        return ""

    priority_text: list[str] = []
    other_text: list[str] = []
    # Only top-level sections: nested <sec> (e.g. "Participants" inside
    # "Methods") are already covered by itertext() on their parent, and
    # walking `.//sec` would duplicate that text.
    for sec in body.findall("sec"):
        all_titles = " ".join(t.text or "" for t in sec.findall(".//title"))
        section_text = "".join(sec.itertext()).strip()
        if any(kw in all_titles.lower() for kw in SECTION_TITLE_KEYWORDS):
            priority_text.append(section_text)
        else:
            other_text.append(section_text)

    if not priority_text and not other_text:
        return "".join(body.itertext()).strip()

    return "\n\n".join(priority_text + other_text)
