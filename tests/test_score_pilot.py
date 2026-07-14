import csv

from src.score_pilot import load_gold, score, write_per_doi

DOI = "10.1371/journal.pone.0000001"


def gold_row(**overrides):
    row = {
        "doi": DOI,
        "sample_id": 1,
        "gender_reported": 1,
        "gender_pct": {"male": 45, "female": 55, "other": 0},
        "race_reported": 0,
        "race_pct": {},
        "education_reported": 0,
        "education_pct": {},
        "ses_reported": 0,
        "ses_value": None,
    }
    row.update(overrides)
    return row


def model_payload(rows, status="ok"):
    return {DOI: {"doi": DOI, "status": status, "rows": rows}}


def test_perfect_match_scores_full_marks():
    gold = [gold_row()]
    model = model_payload([gold_row()])
    per_doi, summary = score(gold, model)
    assert summary["reporting_flag_accuracy"] == 1.0
    assert summary["numeric_accuracy"] == 1.0
    assert summary["sample_count_match_rate"] == 1.0
    assert summary["schema_ok_rate"] == 1.0
    assert per_doi[0]["sample_count_match"] is True


def test_flipped_flag_and_wrong_pct_are_penalized():
    gold = [gold_row(race_reported=1, race_pct={"white": 100})]
    # gender flag flipped (1 -> 0); race pct wrong (100 -> 50).
    model = model_payload(
        [gold_row(gender_reported=0, race_reported=1, race_pct={"white": 50})]
    )
    per_doi, summary = score(gold, model)
    # 3 of 4 reporting flags correct (gender wrong).
    assert summary["reporting_flag_accuracy"] == 0.75
    # Only race is numerically comparable (gender no longer both-reported); wrong.
    assert summary["numeric_accuracy"] == 0.0


def test_missing_model_output_is_total_miss():
    gold = [gold_row()]
    per_doi, summary = score(gold, {})
    assert per_doi[0]["schema_ok"] is False
    assert per_doi[0]["sample_count_match"] is False
    assert per_doi[0]["reporting_flag_accuracy"] == 0.0
    assert "no model output" in per_doi[0]["notes"]
    assert summary["schema_ok_rate"] == 0.0


def test_multi_sample_mismatch_flagged():
    gold = [gold_row(sample_id=1), gold_row(sample_id=2)]
    model = model_payload([gold_row(sample_id=1)])  # model missed sample 2
    per_doi, summary = score(gold, model)
    assert per_doi[0]["sample_count_match"] is False
    assert summary["sample_count_match_rate"] == 0.0


def test_load_gold_parses_pct_json_and_flags(tmp_path):
    gold_csv = tmp_path / "gold.csv"
    with open(gold_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "doi", "sample_id", "gender_reported", "gender_pct",
                "race_reported", "race_pct", "education_reported", "education_pct",
                "ses_reported", "ses_value",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "doi": DOI, "sample_id": "1",
                "gender_reported": "1", "gender_pct": '{"male": 40, "female": 60}',
                "race_reported": "0", "race_pct": "{}",
                "education_reported": "0", "education_pct": "{}",
                "ses_reported": "0", "ses_value": "",
            }
        )
    rows = load_gold(str(gold_csv))
    assert rows[0]["gender_reported"] == 1
    assert rows[0]["gender_pct"] == {"male": 40, "female": 60}
    assert rows[0]["ses_value"] is None


def test_write_per_doi_roundtrips(tmp_path):
    gold = [gold_row()]
    per_doi, _ = score(gold, model_payload([gold_row()]))
    out = tmp_path / "acc.csv"
    write_per_doi(per_doi, str(out))
    with open(out, newline="", encoding="utf-8") as f:
        loaded = list(csv.DictReader(f))
    assert loaded[0]["doi"] == DOI
