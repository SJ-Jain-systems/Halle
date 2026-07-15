import csv
import json
import os
import re

from src.extract_demographics import parse_and_validate
from src.model_backend import EchoModelClient, build_client
from src.run_pilot import run_pilot

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_article.xml")
DOI = "10.1371/journal.pone.0012345"

_DOI_RE = re.compile(r"Article DOI:\s*(\S+)")


def _valid_row(doi):
    return {
        "doi": doi,
        "sample_id": 1,
        "gender_reported": 0,
        "gender_pct": {},
        "race_reported": 0,
        "race_pct": {},
        "education_reported": 0,
        "education_pct": {},
        "ses_reported": 0,
        "ses_value": None,
    }


class CountingClient:
    """Pulls the DOI out of the prompt and echoes back a valid row. Keeps a list
    of every generate() call, and can be told to blow up for certain DOIs."""

    def __init__(self, model_id, raise_for=()):
        self.model_id = model_id
        self.raise_for = set(raise_for)
        self.calls = []

    def generate(self, prompt: str) -> str:
        doi = _DOI_RE.search(prompt).group(1)
        self.calls.append(doi)
        if doi in self.raise_for:
            raise RuntimeError("simulated backend failure")
        return json.dumps([_valid_row(doi)])


def _write_sample_csv(path, dois=(DOI,)):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doi", "subfield", "xml_path"])
        writer.writeheader()
        for doi in dois:
            writer.writerow({"doi": doi, "subfield": "Social psychology", "xml_path": FIXTURE})


MODEL = "meta-llama/Llama-3.3-70B-Instruct"


def _result_path(results_dir, doi):
    return results_dir / MODEL.replace("/", "__") / f"{doi.replace('/', '_')}.json"


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


def test_run_pilot_writes_one_result_file_per_article(tmp_path):
    sample_csv = tmp_path / "sampled_articles.csv"
    _write_sample_csv(sample_csv)
    results_dir = tmp_path / "results"

    run_pilot(
        str(sample_csv),
        model_id="meta-llama/Llama-3.3-70B-Instruct",
        client_factory=FakeClient,
        results_dir=str(results_dir),
    )

    out_file = results_dir / "meta-llama__Llama-3.3-70B-Instruct" / f"{DOI.replace('/', '_')}.json"
    assert out_file.exists()
    payload = json.loads(out_file.read_text())
    assert payload["status"] == "ok"
    assert payload["rows"][0]["doi"] == DOI


def test_resume_skips_already_ok_articles(tmp_path):
    sample_csv = tmp_path / "sampled_articles.csv"
    _write_sample_csv(sample_csv)
    results_dir = tmp_path / "results"

    first = CountingClient(MODEL)
    run_pilot(str(sample_csv), MODEL, client_factory=lambda m: first, results_dir=str(results_dir))
    assert first.calls == [DOI]

    # Second run: the article is already "ok", so we shouldn't call generate()
    # again (and we shouldn't even build the client).
    second = CountingClient(MODEL)
    run_pilot(str(sample_csv), MODEL, client_factory=lambda m: second, results_dir=str(results_dir))
    assert second.calls == []
    assert json.loads(_result_path(results_dir, DOI).read_text())["status"] == "ok"


def test_overwrite_forces_rerun(tmp_path):
    sample_csv = tmp_path / "sampled_articles.csv"
    _write_sample_csv(sample_csv)
    results_dir = tmp_path / "results"

    run_pilot(str(sample_csv), MODEL, client_factory=lambda m: CountingClient(MODEL), results_dir=str(results_dir))

    rerun = CountingClient(MODEL)
    run_pilot(
        str(sample_csv), MODEL,
        client_factory=lambda m: rerun, results_dir=str(results_dir), overwrite=True,
    )
    assert rerun.calls == [DOI]


def test_generic_error_is_recorded_and_run_continues(tmp_path):
    d1, d2 = "10.1371/journal.pone.0000001", "10.1371/journal.pone.0000002"
    sample_csv = tmp_path / "sampled_articles.csv"
    _write_sample_csv(sample_csv, dois=[d1, d2])
    results_dir = tmp_path / "results"

    # d1 blows up with a non-validation error. The run should keep going and
    # still do d2.
    client = CountingClient(MODEL, raise_for=[d1])
    run_pilot(str(sample_csv), MODEL, client_factory=lambda m: client, results_dir=str(results_dir))

    assert json.loads(_result_path(results_dir, d1).read_text())["status"] == "error"
    assert json.loads(_result_path(results_dir, d2).read_text())["status"] == "ok"


def test_echo_backend_generates_schema_valid_output():
    doi = "10.1371/journal.pone.0000042"
    client = build_client("any-model", backend="echo")
    assert isinstance(client, EchoModelClient)
    raw = client.generate(f"...\nArticle DOI: {doi}\n\nArticle text:\nblah")
    rows = parse_and_validate(raw, expected_doi=doi)  # would raise if malformed
    assert rows[0]["doi"] == doi
    assert rows[0]["gender_reported"] == 0


def test_run_pilot_end_to_end_with_echo_backend(tmp_path):
    sample_csv = tmp_path / "sampled_articles.csv"
    _write_sample_csv(sample_csv)
    results_dir = tmp_path / "results"

    run_pilot(
        str(sample_csv), MODEL,
        client_factory=lambda m: build_client(m, backend="echo"), results_dir=str(results_dir),
    )
    payload = json.loads(_result_path(results_dir, DOI).read_text())
    assert payload["status"] == "ok"
    assert payload["rows"][0]["doi"] == DOI
