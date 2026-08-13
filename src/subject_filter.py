"""Filter out non-human-subject articles, so animal studies never reach the
manual-encoding sample or the LLM test set.

The demographic-representativeness study only makes sense for *human* studies:
an article on rats, salmon, wombats or *Drosophila* has no gender / race /
education / SES to encode, and a coder who is handed one either wastes time or
(worse) logs zeros that pollute the trend analysis and any LLM benchmark built
on the same rows.

Two independent layers, because neither is sufficient alone:

1. **Subject taxonomy** (``non_human_subject_reason``). PLOS usually tags a
   non-human study with a term from the ``Organisms > Animals`` branch (or
   ``Model organisms`` / ``Zoology``), which a human study doesn't carry. Cheap
   and taxonomy-driven, matching how the pipeline treats subfields. *But* PLOS
   tagging is inconsistent — some genuine animal studies (e.g. a mouse
   cognition paper filed only under "Cognitive psychology") carry **no**
   organism tag at all, so this layer misses them.

2. **Article text** (``animal_text_reason``). A scan of the title + Methods
   for high-precision animal-model markers — plural/'-ine' animal words
   (``mice``, ``murine``, ``rodents``), lab strain names (``C57BL/6``,
   ``Sprague-Dawley``, ``Wistar``), Latin binomials, non-human-primate terms,
   and animal-ethics-committee statements (``IACUC`` / "Institutional Animal
   Care and Use Committee" — the animal analogue of a human IRB statement).
   These catch the studies layer 1 misses.

Guardrails against dropping real human studies:

* Humans are, taxonomically, animals too. PLOS tags human-subjects work with
  ``Humans`` / ``Homo sapiens``; an explicit human marker *vetoes* taxonomy
  exclusion, so a human/ape comparison whose human data we want is kept.
* Taxonomy matching is exact-term (case-insensitive), never substring, so
  "Human factors" or "Humanities" can't trip the animal set.
* The text layer deliberately omits low-precision cues: bare "mouse" (a
  *computer* mouse in cognitive tasks) and "animal model(s)" (an intro phrase
  in human papers that cite animal literature). Only markers that essentially
  never appear in a human-subjects psychology paper are used.
"""
from __future__ import annotations

import re

# Exact taxonomy terms (lower-cased) that mark a study's subject as a
# non-human organism. Grouped by PLOS thesaurus branch for readability; the
# code flattens them into one lookup set.
_ANIMAL_BRANCH_TERMS: frozenset[str] = frozenset(
    t.lower()
    for t in (
        # Organisms > (Eukaryota >) Animals — the top of the branch. Any animal
        # article carries at least one of these.
        "Animals",
        "Vertebrates",
        "Invertebrates",
        "Amniotes",
        "Metazoa",
        # Vertebrate classes / common groups
        "Mammals",
        "Birds",
        "Fish",
        "Fishes",
        "Amphibians",
        "Reptiles",
        "Rodents",
        "Primates",  # non-human primates; human work is vetoed by HUMAN terms
        "Carnivores",
        "Ungulates",
        "Marsupials",
        "Cetaceans",
        # Invertebrate groups
        "Insects",
        "Arthropoda",
        "Arachnids",
        "Molluscs",
        "Mollusks",
        "Crustaceans",
        "Nematoda",
        "Nematodes",
        "Annelids",
        "Cnidaria",
        "Platyhelminthes",  # flatworms, e.g. planaria
        "Zooplankton",
        # Zoology and its sub-disciplines
        "Zoology",
        "Animal behavior",
        "Animal behaviour",
        "Ethology",
        "Entomology",
        "Ornithology",
        "Ichthyology",
        "Mammalogy",
        "Herpetology",
        "Primatology",
        "Veterinary science",
        "Veterinary medicine",
        "Veterinary medicine and science",
        # Model organisms — the "animal model" studies the encoders flagged.
        "Model organisms",
        "Animal models",
        "Animal models of disease",
        "Mouse models",
        "Rat models",
        "Mice",
        "Rats",
        "Drosophila",
        "Drosophila melanogaster",
        "Zebrafish",
        "Danio rerio",
        "Caenorhabditis elegans",
        "Xenopus",
        # Latin binomials / genera PLOS often uses as the model-organism leaf
        # tag instead of the common name — a real gap: an article tagged only
        # "Mus musculus" (not "Mice"/"Rodents") would otherwise pass.
        "Mus musculus",
        "Mus",
        "Rattus norvegicus",
        "Rattus",
        "Macaca mulatta",
        "Macaca",
        "Macaques",
        "Gallus gallus",
        "Sus scrofa",
        "Bos taurus",
        "Canis",
        "Felis",
        # More vertebrate/invertebrate class-level and common taxa.
        "Rodentia",
        "Muridae",
        "Aves",
        "Mammalia",
        "Actinopterygii",
        "Teleostei",
        "Amphibia",
        "Reptilia",
        "Chordata",
        "Non-human primates",
        "Nonhuman primates",
        "Marine mammals",
        "Bats",
        "Sea lions",
        # A few named taxa that show up as leaf tags in the pilot set, so an
        # article tagged only at the species level is still caught even if its
        # parent branch term is somehow absent.
        "Salmon",
        "Junco",
        "Bonobo",
        "Bonobos",
        "Pan paniscus",
        "Wombats",
        "Planarians",
        "Songbirds",
    )
)

# Explicit human markers. Presence of any of these vetoes exclusion: the
# article is a human study (possibly alongside animal comparisons whose human
# arm we still want to encode).
_HUMAN_TERMS: frozenset[str] = frozenset(
    t.lower()
    for t in (
        "Humans",
        "Human",
        "Homo sapiens",
    )
)

# High-precision animal-model markers for the article-text layer. Each is
# chosen to essentially never appear in a human-subjects psychology paper.
# Deliberately excluded: bare "mouse" (computer mouse in cognitive tasks) and
# "animal model(s)" (human papers cite animal-model literature in the intro).
# `(label, pattern)` — the label is what the reason string reports.
_ANIMAL_TEXT_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = tuple(
    (label, re.compile(pat, re.IGNORECASE))
    for label, pat in (
        ("mice", r"\bmice\b"),
        ("murine", r"\bmurine\b"),
        ("rodent", r"\brodents?\b"),
        ("rats", r"\brats\b"),
        ("C57BL", r"\bc57bl\b"),
        ("BALB/c", r"\bbalb/?c\b"),
        ("Sprague-Dawley", r"\bsprague[-\s]dawley\b"),
        ("Wistar", r"\bwistar\b"),
        ("knockout mice", r"\bknock-?out mice\b"),
        ("transgenic mice", r"\btransgenic mice\b"),
        ("Mus musculus", r"\bmus musculus\b"),
        ("Rattus", r"\brattus\b"),
        ("zebrafish", r"\bzebrafish\b"),
        ("Danio rerio", r"\bdanio rerio\b"),
        ("Drosophila", r"\bdrosophila\b"),
        ("fruit fly", r"\bfruit fl(?:y|ies)\b"),
        ("Xenopus", r"\bxenopus\b"),
        ("C. elegans", r"\b(?:caenorhabditis|c\.?\s?elegans)\b"),
        ("macaque", r"\bmacaques?\b"),
        ("non-human primate", r"\bnon[-\s]?human primates?\b"),
        ("IACUC", r"\biacuc\b"),
        ("institutional animal care", r"\binstitutional animal care\b"),
        ("animal care and use committee", r"\banimal care and use committee\b"),
        ("animal ethics committee", r"\banimal (?:research )?ethics committee\b"),
        ("euthanized/sacrificed", r"\bwere (?:euthan(?:ized|ised)|sacrificed|perfused|decapitated)\b"),
    )
)


def _normalize(subjects: object) -> set[str]:
    """Return the full set of individual taxonomy *terms* (lower-cased), from
    whatever shape the caller has.

    Two producers feed this, and they store subjects differently:

    * ``src/build_corpus_index.py`` (JATS XML path) passes a list of individual
      terms, e.g. ``["Animals", "Vertebrates", "Fish"]``.
    * ``src/build_index_solr.py`` (the Solr discovery path actually run on
      Rivanna) stores the ``subject`` column as ';'-joined **slash-delimited
      paths**, e.g.
      ``"/Biology and life sciences/Organisms/Animals/Vertebrates/Fish"``.

    So we split on both ``;`` (multiple subjects) and ``/`` (path segments
    within one subject) — the same segmentation ``solr_client`` uses — which
    turns a path into its component terms and leaves an already-split list of
    terms unchanged. Without the ``/`` split an entire Solr path would be one
    unmatchable "term" and every animal study would pass straight through.
    """
    if subjects is None:
        return set()
    if isinstance(subjects, str):
        raw = subjects.split(";")
    else:
        raw = list(subjects)
    terms: set[str] = set()
    for item in raw:
        for segment in str(item).split("/"):
            segment = segment.strip().lower()
            if segment:
                terms.add(segment)
    return terms


def non_human_subject_reason(subjects: object) -> str | None:
    """Return a human-readable reason string if this article's subject
    taxonomy marks it as a non-human (animal / model-organism) study, else
    ``None``.

    ``subjects`` may be a list of taxonomy terms (``ArticleMetadata.subject``)
    or a ';'-joined string of terms or slash-delimited Solr paths (the
    ``subject`` column written by either index builder) — see ``_normalize``.

    An explicit human marker (``Humans`` / ``Homo sapiens``) always wins, so a
    study that includes human participants is never excluded even if it also
    references an animal.
    """
    terms = _normalize(subjects)
    if terms & _HUMAN_TERMS:
        return None
    hits = terms & _ANIMAL_BRANCH_TERMS
    if hits:
        return "non-human subject taxonomy: " + ", ".join(sorted(hits))
    return None


def animal_text_reason(text: str | None) -> str | None:
    """Return a reason string if the article *text* contains a high-precision
    animal-model marker, else ``None``.

    This is the second layer: it catches animal studies that PLOS left without
    an organism subject tag (so ``non_human_subject_reason`` can't see them).
    Pass the title + Methods/Participants text — see
    ``jats_xml.get_screening_text`` — not the whole body, to keep the scan
    focused on where the study describes its subjects.

    Every pattern in ``_ANIMAL_TEXT_PATTERNS`` is one that essentially never
    appears in a human-subjects psychology paper; the reason names the markers
    found so a reviewer can sanity-check any exclusion.
    """
    if not text:
        return None
    hits = [label for label, pat in _ANIMAL_TEXT_PATTERNS if pat.search(text)]
    if hits:
        return "animal-model markers in text: " + ", ".join(hits)
    return None


def non_human_reason(subjects: object = None, text: str | None = None) -> str | None:
    """Combined screen: return the first reason this article is non-human, from
    either the subject taxonomy (layer 1) or the article text (layer 2), else
    ``None``. Callers with only one signal available can pass just that one."""
    return non_human_subject_reason(subjects) or animal_text_reason(text)


def is_human_subject_study(subjects: object = None, text: str | None = None) -> bool:
    """Convenience inverse of :func:`non_human_reason`: ``True`` when neither
    layer flags the article as non-human."""
    return non_human_reason(subjects, text) is None
