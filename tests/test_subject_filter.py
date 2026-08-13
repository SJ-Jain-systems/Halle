import pytest

from src.subject_filter import (
    is_human_subject_study,
    non_human_subject_reason,
)

# Subject-tag sets modelled on the animal-model articles the pilot coders
# flagged (Manual_Encoding_1: salmon, junco, mouse, rats, bonobos, planaria,
# Drosophila, wombat, ...). PLOS carries the full Organisms/Zoology branch on
# each, alongside the Psychology subfield.
ANIMAL_SUBJECTS = [
    ["Biology and life sciences", "Psychology", "Behavior",
     "Zoology", "Animal behavior", "Organisms", "Animals", "Vertebrates", "Fish"],
    ["Biology and life sciences", "Psychology", "Behavior",
     "Organisms", "Animals", "Vertebrates", "Birds"],            # junco
    ["Biology and life sciences", "Psychology", "Behavior",
     "Model organisms", "Animal models", "Mouse models", "Mice"],  # rodent model
    ["Biology and life sciences", "Psychology", "Instinct",
     "Organisms", "Animals", "Vertebrates", "Mammals", "Rodents"],  # rats
    ["Biology and life sciences", "Psychology",
     "Organisms", "Animals", "Vertebrates", "Mammals", "Primates"],  # bonobos
    ["Biology and life sciences", "Psychology", "Behavior",
     "Organisms", "Animals", "Invertebrates", "Platyhelminthes"],   # planaria
    ["Biology and life sciences", "Psychology",
     "Model organisms", "Drosophila melanogaster"],                 # Drosophila
]

# Human psychology studies — the ones we must keep. None carry an Animals /
# Zoology / Model-organisms tag.
HUMAN_SUBJECTS = [
    ["Biology and life sciences", "Psychology", "Social psychology",
     "Medicine and health sciences", "People and places", "Population groupings"],
    ["Biology and life sciences", "Psychology", "Cognitive psychology",
     "Social sciences"],
    ["Medicine and health sciences", "Mental health and psychiatry",
     "Biology and life sciences", "Psychology", "Clinical psychology"],
]


@pytest.mark.parametrize("subjects", ANIMAL_SUBJECTS)
def test_animal_studies_are_flagged(subjects):
    reason = non_human_subject_reason(subjects)
    assert reason is not None
    assert not is_human_subject_study(subjects)


@pytest.mark.parametrize("subjects", HUMAN_SUBJECTS)
def test_human_studies_are_kept(subjects):
    assert non_human_subject_reason(subjects) is None
    assert is_human_subject_study(subjects)


def test_explicit_human_tag_vetoes_animal_terms():
    # A human/ape comparison whose human arm we still want: tagged both
    # Primates and Humans -> kept.
    subjects = ["Biology and life sciences", "Psychology", "Primates", "Humans"]
    assert non_human_subject_reason(subjects) is None


def test_matching_is_case_insensitive():
    assert non_human_subject_reason(["ANIMALS", "vertebrates"]) is not None


def test_matching_is_exact_term_not_substring():
    # "Human factors" / "Humanities" must not register as the animal term set,
    # and unrelated words containing an animal substring must not false-match.
    assert non_human_subject_reason(["Human factors engineering"]) is None
    assert non_human_subject_reason(["Ratings", "Animalia studies"]) is None


def test_accepts_semicolon_joined_string():
    # The corpus-index `subject` column stores terms ';'-joined; the filter
    # must accept that form directly.
    joined = "Biology and life sciences;Psychology;Behavior;Organisms;Animals;Vertebrates;Fish"
    assert non_human_subject_reason(joined) is not None
    assert non_human_subject_reason("Psychology;Social psychology") is None


def test_accepts_solr_slash_delimited_paths():
    # build_index_solr.py (the path actually run on Rivanna) stores the subject
    # column as ';'-joined *slash-delimited* Solr paths. The filter must split
    # path segments into terms, or every animal study slips through.
    animal = (
        "/Biology and life sciences/Psychology/Behavior;"
        "/Biology and life sciences/Zoology/Animal behavior;"
        "/Biology and life sciences/Organisms/Animals/Vertebrates/Fish"
    )
    reason = non_human_subject_reason(animal)
    assert reason is not None
    assert "fish" in reason

    human = (
        "/Biology and life sciences/Psychology/Social psychology;"
        "/Medicine and health sciences/Mental health and psychiatry"
    )
    assert non_human_subject_reason(human) is None

    # Human veto still applies when 'Humans' appears as a path leaf.
    veto = (
        "/Biology and life sciences/Organisms/Animals/Vertebrates/Mammals/Primates;"
        "/Biology and life sciences/Organisms/Eukaryota/Animals/Vertebrates/Humans"
    )
    assert non_human_subject_reason(veto) is None

    # A single path passed as a one-element list (not ';'-joined) also works.
    assert non_human_subject_reason(
        ["/Biology and life sciences/Organisms/Animals/Invertebrates/Insects"]
    ) is not None


def test_empty_or_missing_subjects_is_not_flagged():
    assert non_human_subject_reason(None) is None
    assert non_human_subject_reason([]) is None
    assert non_human_subject_reason("") is None


def test_reason_names_the_matched_terms():
    reason = non_human_subject_reason(["Organisms", "Animals", "Vertebrates", "Fish"])
    assert "animals" in reason
    assert "fish" in reason
