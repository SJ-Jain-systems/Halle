from src.sample_articles import stratified_sample

FAKE_SUBFIELDS = ["Social psychology", "Cognitive psychology"]


def fake_fetch(subfield: str) -> list[dict]:
    return [
        {"id": f"10.1371/journal.pone.{subfield[:3]}{i}", "title": f"{subfield} article {i}"}
        for i in range(5)
    ]


def test_stratified_sample_returns_per_subfield_count():
    result = stratified_sample(subfields=FAKE_SUBFIELDS, per_subfield=2, fetch_fn=fake_fetch)
    assert len(result) == 4
    counts = {}
    for row in result:
        counts[row["subfield"]] = counts.get(row["subfield"], 0) + 1
    assert counts == {"Social psychology": 2, "Cognitive psychology": 2}


def test_stratified_sample_is_reproducible_with_seed():
    first = stratified_sample(subfields=FAKE_SUBFIELDS, per_subfield=2, seed=7, fetch_fn=fake_fetch)
    second = stratified_sample(subfields=FAKE_SUBFIELDS, per_subfield=2, seed=7, fetch_fn=fake_fetch)
    assert [row["id"] for row in first] == [row["id"] for row in second]


def test_stratified_sample_raises_when_not_enough_candidates():
    def sparse_fetch(subfield: str) -> list[dict]:
        return [{"id": "only-one"}]

    try:
        stratified_sample(subfields=FAKE_SUBFIELDS, per_subfield=2, fetch_fn=sparse_fetch)
        assert False, "expected ValueError"
    except ValueError:
        pass
