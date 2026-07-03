import pytest

from src.sample_articles import distinct_subfields, stratified_sample

FAKE_SUBFIELDS = ["Social psychology", "Cognitive psychology"]


def fake_rows() -> list[dict]:
    rows = []
    for subfield in FAKE_SUBFIELDS:
        for i in range(5):
            rows.append(
                {
                    "doi": f"10.1371/journal.pone.{subfield[:3]}{i}",
                    "title": f"{subfield} article {i}",
                    "matched_subfields": subfield,
                }
            )
    return rows


def test_stratified_sample_returns_per_subfield_count():
    result = stratified_sample(fake_rows(), subfields=FAKE_SUBFIELDS, per_subfield=2)
    assert len(result) == 4
    counts = {}
    for row in result:
        counts[row["subfield"]] = counts.get(row["subfield"], 0) + 1
    assert counts == {"Social psychology": 2, "Cognitive psychology": 2}


def test_stratified_sample_is_reproducible_with_seed():
    rows = fake_rows()
    first = stratified_sample(rows, subfields=FAKE_SUBFIELDS, per_subfield=2, seed=7)
    second = stratified_sample(rows, subfields=FAKE_SUBFIELDS, per_subfield=2, seed=7)
    assert [row["doi"] for row in first] == [row["doi"] for row in second]


def test_stratified_sample_raises_when_not_enough_candidates():
    sparse_rows = [{"doi": "only-one", "matched_subfields": "Social psychology"}]
    with pytest.raises(ValueError):
        stratified_sample(sparse_rows, subfields=FAKE_SUBFIELDS, per_subfield=2)


def test_stratified_sample_only_matches_exact_subfield_token():
    # "Social psychology" should not accidentally match a row tagged only
    # "Cognitive psychology" via substring overlap.
    rows = [
        {"doi": "a", "matched_subfields": "Cognitive psychology"},
        {"doi": "b", "matched_subfields": "Social psychology"},
        {"doi": "c", "matched_subfields": "Social psychology"},
    ]
    result = stratified_sample(rows, subfields=["Social psychology"], per_subfield=2)
    assert {r["doi"] for r in result} == {"b", "c"}


def test_distinct_subfields_collects_all_present():
    rows = [
        {"doi": "a", "matched_subfields": "Social psychology;Cognitive psychology"},
        {"doi": "b", "matched_subfields": "Clinical psychology"},
        {"doi": "c", "matched_subfields": "Social psychology"},
    ]
    assert set(distinct_subfields(rows)) == {
        "Social psychology", "Cognitive psychology", "Clinical psychology"
    }


def test_stratified_sample_auto_covers_all_subfields_and_skips_sparse():
    # Auto mode (subfields=None): stratify over every subfield present, but
    # silently skip ones without enough articles instead of erroring.
    rows = (
        [{"doi": f"soc{i}", "matched_subfields": "Social psychology"} for i in range(3)]
        + [{"doi": f"cog{i}", "matched_subfields": "Cognitive psychology"} for i in range(3)]
        + [{"doi": "rare0", "matched_subfields": "Psychometrics"}]  # only 1, too few
    )
    result = stratified_sample(rows, per_subfield=2)
    covered = {r["subfield"] for r in result}
    assert covered == {"Social psychology", "Cognitive psychology"}
    assert len(result) == 4
