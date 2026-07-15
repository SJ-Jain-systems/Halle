import csv

from src.score_pilot import (
    confusion_counts,
    load_gold,
    metrics_from_counts,
    score,
    write_metrics,
    write_per_doi,
)

DOI = "10.1371/journal.pone.0000001"


def gold_row(**overrides):
    row = {
        "doi": DOI,
        "sample_id": 1,
        "gender": {"reported": 1, "pct": {"male": 45, "female": 55, "other": 0}},
        "race": {"reported": 0, "pct": {}},
        "education": {"reported": 0, "pct": {}},
        "ses": {"reported": 0, "value": None},
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
    gold = [gold_row(race={"reported": 1, "pct": {"white": 100}})]
    # gender flag flipped (1 -> 0); race pct wrong (100 -> 50).
    model = model_payload(
        [gold_row(
            gender={"reported": 0, "pct": {}},
            race={"reported": 1, "pct": {"white": 50}},
        )]
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


def test_load_gold_parses_combined_columns(tmp_path):
    gold_csv = tmp_path / "gold.csv"
    with open(gold_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["doi", "sample_id", "gender", "race", "education", "ses"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "doi": DOI, "sample_id": "1",
                "gender": "1, 40% Male, 60% Female",
                "race": "0", "education": "0", "ses": "0",
            }
        )
    rows = load_gold(str(gold_csv))
    assert rows[0]["gender"] == {"reported": 1, "pct": {"male": 40, "female": 60}}
    assert rows[0]["ses"] == {"reported": 0, "value": None}


def test_confusion_and_metrics_perfect_match_pass_gate():
    # default gold_row: gender reported=1, race/education/ses reported=0
    gold = [gold_row()]
    counts = confusion_counts(gold, model_payload([gold_row()]))
    assert counts["gender"] == {"tp": 1, "fp": 0, "fn": 0, "tn": 0}
    assert counts["race"] == {"tp": 0, "fp": 0, "fn": 0, "tn": 1}
    metrics = metrics_from_counts(counts)
    assert metrics["overall"]["recall"] == 1.0
    assert metrics["overall"]["precision"] == 1.0
    assert metrics["overall"]["accuracy"] == 1.0
    assert metrics["overall"]["passed"] is True


def test_missed_report_is_false_negative_and_fails_recall():
    gold = [gold_row(gender={"reported": 1, "pct": {"male": 50, "female": 50}})]
    model = model_payload([gold_row(gender={"reported": 0, "pct": {}})])
    counts = confusion_counts(gold, model)
    assert counts["gender"]["fn"] == 1
    metrics = metrics_from_counts(counts)
    assert metrics["gender"]["recall"] == 0.0
    assert metrics["gender"]["recall_pass"] is False
    assert metrics["overall"]["passed"] is False


def test_hallucinated_report_is_false_positive_and_fails_precision():
    gold = [gold_row(ses={"reported": 0, "value": None})]
    model = model_payload([gold_row(ses={"reported": 1, "value": None})])
    counts = confusion_counts(gold, model)
    assert counts["ses"]["fp"] == 1
    metrics = metrics_from_counts(counts)
    assert metrics["ses"]["precision"] == 0.0


def test_ses_02_scale_collapses_to_reported():
    # SES reported as a numeric threshold (2) still counts as "reported" (>=1).
    gold = [gold_row(ses={"reported": 2, "value": 30000})]
    model = model_payload([gold_row(ses={"reported": 2, "value": 30000})])
    counts = confusion_counts(gold, model)
    assert counts["ses"]["tp"] == 1


def test_missing_model_output_counts_reported_vars_as_false_negatives():
    gold = [gold_row()]  # gender reported
    counts = confusion_counts(gold, {})
    assert counts["gender"]["fn"] == 1
    assert counts["race"]["tn"] == 1


def test_summary_carries_gate_and_score_passes_thresholds():
    _, summary = score([gold_row()], model_payload([gold_row()]))
    assert summary["passed"] is True
    assert "overall" in summary["metrics"]
    assert summary["metrics"]["gender"]["recall"] == 1.0


def test_write_metrics_roundtrips(tmp_path):
    _, summary = score([gold_row()], model_payload([gold_row()]))
    out = tmp_path / "metrics.csv"
    write_metrics(summary["metrics"], str(out))
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    variables = {r["variable"] for r in rows}
    assert {"gender", "race", "education", "ses", "overall"} <= variables


def test_write_per_doi_roundtrips(tmp_path):
    gold = [gold_row()]
    per_doi, _ = score(gold, model_payload([gold_row()]))
    out = tmp_path / "acc.csv"
    write_per_doi(per_doi, str(out))
    with open(out, newline="", encoding="utf-8") as f:
        loaded = list(csv.DictReader(f))
    assert loaded[0]["doi"] == DOI
