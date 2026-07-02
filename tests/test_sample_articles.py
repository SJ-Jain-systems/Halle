import pytest

from src.sample_articles import stratified_sample

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
