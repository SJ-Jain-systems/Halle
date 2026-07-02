import csv
import json
import os

import pandas as pd

from src.analyze_trends import run

FIELDS = [
    "doi", "sample_id", "subfield", "matched_subfields", "subject_level_1", "subject",
    "publication_date", "year", "stage", "lead_institution",
    "gender_reported", "gender_pct", "race_reported", "race_pct",
    "education_reported", "education_pct", "ses_reported", "ses_value",
]

ROWS = [
    {
        "doi": "10.1371/journal.pone.0000001", "sample_id": 1, "subfield": "Social psychology",
        "matched_subfields": "Social psychology", "subject_level_1": "", "subject": "",
        "publication_date": "2016-01-01", "year": 2016, "stage": "middle", "lead_institution": "UVA",
        "gender_reported": 1, "gender_pct": json.dumps({"male": 40, "female": 60}),
        "race_reported": 0, "race_pct": json.dumps({}),
        "education_reported": 0, "education_pct": json.dumps({}),
        "ses_reported": 0, "ses_value": "",
    },
    {
        "doi": "10.1371/journal.pone.0000002", "sample_id": 1, "subfield": "Cognitive psychology",
        "matched_subfields": "Cognitive psychology", "subject_level_1": "", "subject": "",
        "publication_date": "2016-06-01", "year": 2016, "stage": "middle", "lead_institution": "UVA",
        "gender_reported": 0, "gender_pct": json.dumps({}),
        "race_reported": 0, "race_pct": json.dumps({}),
        "education_reported": 0, "education_pct": json.dumps({}),
        "ses_reported": 0, "ses_value": "",
    },
    {
        "doi": "10.1371/journal.pone.0000003", "sample_id": 1, "subfield": "Clinical psychology",
        "matched_subfields": "Clinical psychology", "subject_level_1": "", "subject": "",
        "publication_date": "2021-01-01", "year": 2021, "stage": "covid", "lead_institution": "UVA",
        "gender_reported": 1, "gender_pct": json.dumps({"male": 50, "female": 50}),
        "race_reported": 0, "race_pct": json.dumps({}),
        "education_reported": 0, "education_pct": json.dumps({}),
        "ses_reported": 2, "ses_value": "30000",
    },
]


def _write_table(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(ROWS)


def test_run_computes_reporting_rate_by_year(tmp_path):
    table_csv = tmp_path / "demographics_table.csv"
    _write_table(table_csv)
    out_dir = tmp_path / "analysis"

    run(str(table_csv), str(out_dir), make_plots=False)

    rate_df = pd.read_csv(out_dir / "gender_reporting_rate_by_year.csv")
    row_2016 = rate_df[rate_df["year"] == 2016].iloc[0]
    # 1 of 2 samples in 2016 reported gender
    assert row_2016["reporting_rate_pct"] == 50.0

    row_2021 = rate_df[rate_df["year"] == 2021].iloc[0]
    assert row_2021["reporting_rate_pct"] == 100.0


def test_run_computes_mean_subgroup_pct(tmp_path):
    table_csv = tmp_path / "demographics_table.csv"
    _write_table(table_csv)
    out_dir = tmp_path / "analysis"

    run(str(table_csv), str(out_dir), make_plots=False)

    mean_df = pd.read_csv(out_dir / "gender_mean_pct_by_year.csv")
    male_2016 = mean_df[(mean_df["year"] == 2016) & (mean_df["subgroup"] == "male")].iloc[0]
    assert male_2016["mean_pct"] == 40.0


def test_run_writes_ses_reporting_detail(tmp_path):
    table_csv = tmp_path / "demographics_table.csv"
    _write_table(table_csv)
    out_dir = tmp_path / "analysis"

    run(str(table_csv), str(out_dir), make_plots=False)

    assert os.path.exists(out_dir / "ses_reporting_detail_by_year.csv")
    assert os.path.exists(out_dir / "ses_reporting_detail_by_stage.csv")
