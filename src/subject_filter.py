"""Filter out non-human-subject articles by their PLOS subject taxonomy.

The demographic-representativeness study only makes sense for *human* studies:
an article on rats, salmon, wombats or *Drosophila* has no gender / race /
education / SES to encode, and a coder who is handed one either wastes time or
(worse) logs zeros that pollute the trend analysis and any LLM benchmark built
on the same rows. So animal-model articles must be dropped *before* an article
is ever drawn into the manual-encoding sample or the model test set.

This is done "by subject area" — PLOS tags every article with its full subject
taxonomy, and a non-human-animal study always carries at least one term from
the ``Organisms > Animals`` branch (or the ``Model organisms`` / ``Zoology``
branches), while a human study does not. We key off that rather than trying to
read the body text, so the decision is cheap and taxonomy-driven, matching how
the rest of the pipeline treats subfields (``jats_xml.get_psychology_subfields``).

Design notes / guardrails:

* Humans are, taxonomically, animals too. PLOS tags human-subjects work with
  ``Humans`` / ``Homo sapiens`` (and files it under ``People and places`` and
  ``Medicine and health sciences``). So an explicit human marker *vetoes*
  exclusion — an article tagged both ``Primates`` and ``Humans`` (e.g. a
  human/ape comparison whose human data we *do* want) is kept.
* Matching is on the high-level branch terms (``Animals``, ``Vertebrates``,
  ``Invertebrates``, ``Zoology``, ``Animal models``, ...) plus a handful of
  common taxa and named model organisms. Any genuine animal article carries at
  least one of these, so we don't need to enumerate every species; a bare
  species tag we miss is still caught by its ``Animals`` / ``Vertebrates``
  parent term. Matching is exact-term (case-insensitive), not substring, so
  "Human factors" or "Humanities" can never trip the animal set.
"""
from __future__ import annotations

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


def _normalize(subjects: object) -> set[str]:
    """Accept either a list of subject terms or a single ';'-joined string
    (the form stored in the corpus-index ``subject`` column) and return a set
    of lower-cased, stripped terms."""
    if subjects is None:
        return set()
    if isinstance(subjects, str):
        parts = subjects.split(";")
    else:
        parts = list(subjects)
    return {p.strip().lower() for p in parts if p and p.strip()}


def non_human_subject_reason(subjects: object) -> str | None:
    """Return a human-readable reason string if this article's subject
    taxonomy marks it as a non-human (animal / model-organism) study, else
    ``None``.

    ``subjects`` may be a list of taxonomy terms (``ArticleMetadata.subject``)
    or a ';'-joined string (the corpus-index ``subject`` column).

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


def is_human_subject_study(subjects: object) -> bool:
    """Convenience inverse of :func:`non_human_subject_reason`: ``True`` when
    the article is *not* flagged as a non-human study."""
    return non_human_subject_reason(subjects) is None
