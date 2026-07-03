# NOTES
# PLOS articles are stored as XML files. This is the toolbox for pulling things
# out of one: the ID, title, journal, date, subject tags, the corresponding
# author's institution, and the body text.
#
# The one piece the model needs is the body text, and mostly the
# Methods/Participants section, because that's where demographics live.
# get_extraction_text pulls the whole body but moves those sections to the front
# so the model sees them first.
#
# The subject-tag functions were how I originally found psychology papers from
# the files. I moved that job to the search engine because the files are
# missing tags for some years. These still read tags where they exist, and they
# document how the tags are laid out.
#
# If you don't read XML: tree.find(...) means "go dig out this element", and the
# slash paths are addresses telling it where to look.
"""Read metadata and text out of PLOS article XML files."""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

from lxml import etree

# Section headings I treat as probably where the demographics are.
SECTION_TITLE_KEYWORDS = ("method", "participant", "sample", "procedure")


@dataclass
class ArticleMetadata:
    # A tidy box holding everything I pull about one article.
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
    # Load an XML file into memory so the other functions can dig around in it.
    return etree.parse(xml_path)


def get_doi(tree: etree._ElementTree) -> str:
    # The article's permanent ID.
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
    # PLOS records several dates: online, print, and so on. I try them in a
    # sensible order and take the first that gives a usable year. If month or day
    # is missing I default to 1, since I mostly care about the year.
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
    """Read the subject tags.

    subject_level_1 is the top of each branch, like "Biology and life sciences".
    subject is every tag at any depth.

    The startswith("Discipline") check is the fix for the tag-format change. Old
    files say "Discipline", new files say "Discipline-v3". Accept both. Miss this
    and modern papers come back empty.
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
    """Return the subfield(s) sitting directly under a "Psychology" tag.

    Find the "Psychology" tag, then grab whatever is one level under it. That is
    the subfield. This is why I get all 23 subfields instead of a hardcoded
    five: I take whatever PLOS actually filed the paper under. If Psychology has
    no child, I return "Psychology" on its own. Empty if the paper isn't under
    Psychology at all.
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
    """Get the corresponding author's institution.

    Find the corresponding author, follow the cross-reference to their
    affiliation, and read the institution name. Falls back to the first author if
    none is marked as corresponding.
    """
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
    # No structured institution tag, so fall back to the affiliation's full text.
    text = "".join(aff.itertext()).strip()
    return text or None


def parse_metadata(xml_path: str) -> ArticleMetadata:
    # Open a file once and pull everything into one tidy object.
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
    """Get the body text to hand the model, Methods/Participants first.

    I keep the whole body but reorder it so Methods and Participants come first.
    That's where the demographics are, and models pay most attention to the start
    of a long prompt.
    """
    tree = parse_tree(xml_path)
    body = tree.find(".//body")
    if body is None:
        return ""

    priority_text: list[str] = []
    other_text: list[str] = []
    # Top-level sections only. A nested section (like Participants inside
    # Methods) is already covered by its parent, so walking every section would
    # duplicate text.
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
