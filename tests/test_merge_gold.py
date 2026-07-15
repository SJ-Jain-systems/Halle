import csv

from src.merge_gold import (
    cohen_kappa,
    consensus,
    inter_rater_agreement,
    load_coded,
    write_consensus,
)

FIELDS = ["doi", "coder", "sample_id", "gender", "race", "education", "ses"]


def _write_sheet(path, coder, race_by_doi):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for doi, race in race_by_doi.items():
            writer.writerow({
                "doi": doi, "coder": coder, "sample_id": 1,
                "gender": "1, 40% Male, 60% Female", "race": race,
                "education": "0", "ses": "0",
            })


def test_cohen_kappa_perfect_and_chance():
    assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    # All identical single category -> degenerate, treated as full agreement.
    assert cohen_kappa([0, 0, 0], [0, 0, 0]) == 1.0
    # Systematic anti-agreement (each rater the other's opposite) -> negative.
    assert cohen_kappa([1, 0], [0, 1]) < 0


def test_inter_rater_agreement_two_coders(tmp_path):
    a = tmp_path / "alice.csv"
    b = tmp_path / "bob.csv"
    # Agree on d1 and d3, disagree on d2's race.
    _write_sheet(a, "A", {"d1": "1, 100% White", "d2": "1, 50% White, 50% Black", "d3": "0"})
    _write_sheet(b, "B", {"d1": "1, 100% White", "d2": "0", "d3": "0"})

    units, coders = load_coded([str(a), str(b)])
    assert coders == ["A", "B"]
    agreement = inter_rater_agreement(units, coders)
    # gender identical across all three -> perfect
    assert agreement["gender"]["percent_agreement"] == 1.0
    assert agreement["gender"]["cohen_kappa"] == 1.0
    # race: 2 of 3 agree
    assert agreement["race"]["percent_agreement"] == round(2 / 3, 3)


def test_consensus_majority_and_disagreement_flag(tmp_path):
    a = tmp_path / "alice.csv"
    b = tmp_path / "bob.csv"
    _write_sheet(a, "A", {"d1": "1, 100% White", "d2": "1, 50% White, 50% Black"})
    _write_sheet(b, "B", {"d1": "1, 100% White", "d2": "0"})

    units, coders = load_coded([str(a), str(b)])
    rows, disagreements = consensus(units, coders)
    by_doi = {r["doi"]: r for r in rows}
    # Both coders agree on d1 race -> consensus reported.
    assert by_doi["d1"]["race"] == "1, 100% White"
    # Even split on d2 race -> defaults to not reported, and is flagged.
    assert by_doi["d2"]["race"] == "0"
    flagged = {d["doi"] for d in disagreements}
    assert "d2" in flagged and "d1" not in flagged


def test_consensus_median_pct_across_three_coders(tmp_path):
    a, b, c = (tmp_path / f"{n}.csv" for n in ("a", "b", "c"))
    _write_sheet(a, "A", {"d1": "1, 60% White, 40% Black"})
    _write_sheet(b, "B", {"d1": "1, 70% White, 30% Black"})
    _write_sheet(c, "C", {"d1": "1, 80% White, 20% Black"})
    units, coders = load_coded([str(a), str(b), str(c)])
    rows, _ = consensus(units, coders)
    # Majority reported; per-subgroup median (60/70/80 -> 70).
    assert rows[0]["race"] == "1, 70% White, 30% Black"


def test_write_consensus_is_scorable_schema(tmp_path):
    a = tmp_path / "alice.csv"
    _write_sheet(a, "A", {"d1": "1, 100% White"})
    units, coders = load_coded([str(a)])
    rows, _ = consensus(units, coders)
    out = tmp_path / "pilot_gold.csv"
    write_consensus(rows, str(out))

    # score_pilot.load_gold must be able to read the consensus file.
    from src.score_pilot import load_gold
    loaded = load_gold(str(out))
    assert loaded[0]["doi"] == "d1"
    assert loaded[0]["race"] == {"reported": 1, "pct": {"white": 100}}
