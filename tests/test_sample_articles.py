import pytest

from src.sample_articles import (
    distinct_subfields,
    filter_out_non_human,
    stratified_sample,
)

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


def test_filter_out_non_human_drops_animal_rows():
    # Safety net for sampling from an index built before the animal filter
    # existed: rows whose `subject` column marks them as animal studies must be
    # dropped, human rows kept.
    rows = [
        {"doi": "human", "matched_subfields": "Social psychology",
         "subject": "Biology and life sciences;Psychology;Social psychology"},
        {"doi": "salmon", "matched_subfields": "Behavior",
         "subject": "Psychology;Behavior;Organisms;Animals;Vertebrates;Fish"},
        {"doi": "mouse", "matched_subfields": "Behavior",
         "subject": "Psychology;Behavior;Model organisms;Mouse models"},
    ]
    kept = filter_out_non_human(rows)
    assert {r["doi"] for r in kept} == {"human"}


def test_distinct_subfields_collects_all_present():
    rows = [
        {"doi": "a", "matched_subfields": "Social psychology;Cognitive psychology"},
        {"doi": "b", "matched_subfields": "Clinical psychology"},
        {"doi": "c", "matched_subfields": "Social psychology"},
    ]
    assert set(distinct_subfields(rows)) == {
        "Social psychology", "Cognitive psychology", "Clinical psychology"
    }


def _rows_with_sizes(sizes: dict) -> list[dict]:
    rows = []
    for subfield, n in sizes.items():
        for i in range(n):
            rows.append({"doi": f"{subfield[:4]}{i}", "matched_subfields": subfield})
    return rows


def test_stratified_sample_total_size_hits_target_with_floor_and_topup():
    # 4 subfields, unequal sizes; ask for 20 total with a floor of 2 each.
    sizes = {"A": 30, "B": 30, "C": 5, "D": 5}
    rows = _rows_with_sizes(sizes)
    result = stratified_sample(
        rows, subfields=list(sizes), total_size=20, min_per_subfield=2, seed=1
    )
    assert len(result) == 20
    counts = {}
    for r in result:
        counts[r["subfield"]] = counts.get(r["subfield"], 0) + 1
    # Every subfield keeps at least the floor...
    assert min(counts.values()) >= 2
    assert set(counts) == set(sizes)
    # ...and the big subfields get more than the small ones (proportional top-up).
    assert counts["A"] > counts["C"]


def test_stratified_sample_total_size_is_reproducible_with_seed():
    sizes = {"A": 30, "B": 30, "C": 10}
    rows = _rows_with_sizes(sizes)
    first = stratified_sample(rows, subfields=list(sizes), total_size=25, min_per_subfield=2, seed=9)
    second = stratified_sample(rows, subfields=list(sizes), total_size=25, min_per_subfield=2, seed=9)
    assert [r["doi"] for r in first] == [r["doi"] for r in second]


def test_stratified_sample_floor_wins_when_target_below_floor_total():
    # 4 subfields × floor 2 = 8 minimum; a target of 5 can't undercut coverage.
    sizes = {"A": 10, "B": 10, "C": 10, "D": 10}
    rows = _rows_with_sizes(sizes)
    result = stratified_sample(
        rows, subfields=list(sizes), total_size=5, min_per_subfield=2, seed=1
    )
    assert len(result) == 8
    assert {r["subfield"] for r in result} == set(sizes)


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
