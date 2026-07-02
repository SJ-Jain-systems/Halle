import csv
import json
import os

from src.compare_llms import run_comparison

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_article.xml")
DOI = "10.1371/journal.pone.0012345"


class FakeClient:
    def __init__(self, model_id):
        self.model_id = model_id

    def generate(self, prompt: str) -> str:
        return json.dumps(
            [
                {
                    "doi": DOI,
                    "sample_id": 1,
                    "gender_reported": 1,
                    "gender_pct": {"male": 45, "female": 55, "other": 0},
                    "race_reported": 1,
                    "race_pct": {"white": 60, "black": 20, "hispanic": 10, "asian": 5, "other": 5},
                    "education_reported": 0,
                    "education_pct": {},
                    "ses_reported": 0,
                    "ses_value": None,
                }
            ]
        )


def _write_sample_csv(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doi", "subfield", "xml_path"])
        writer.writeheader()
        writer.writerow({"doi": DOI, "subfield": "Social psychology", "xml_path": FIXTURE})


def test_run_comparison_writes_one_result_file_per_model(tmp_path):
    sample_csv = tmp_path / "sampled_articles.csv"
    _write_sample_csv(sample_csv)
    results_dir = tmp_path / "results"

    run_comparison(
        str(sample_csv),
        models=["fake/model-a", "fake/model-b"],
        client_factory=FakeClient,
        results_dir=str(results_dir),
    )

    for model_id in ("fake__model-a", "fake__model-b"):
        out_file = results_dir / model_id / f"{DOI.replace('/', '_')}.json"
        assert out_file.exists()
        payload = json.loads(out_file.read_text())
        assert payload["status"] == "ok"
        assert payload["rows"][0]["doi"] == DOI
