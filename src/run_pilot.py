"""Run the 46-article pilot through the chosen model (docs/DECISIONS.md #3:
meta-llama/Llama-3.3-70B-Instruct) and save output for a QA spot-check
against docs/SCORING_RUBRIC.md before committing to a full-corpus run.

Usage (on a Rivanna GPU node — see docs/RUNNING_ON_RIVANNA.md):
    python -m src.run_pilot --sample data/sampled_articles.csv --backend vllm

Reads full text straight from the local corpus XML (`xml_path` column in
data/sampled_articles.csv) — no network calls.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import yaml

from src.extract_demographics import ExtractionValidationError, extract_demographics_from_xml
from src.model_backend import build_client


def load_sample(csv_path: str) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_pilot(
    sample_csv: str,
    model_id: str,
    client_factory,
    results_dir: str = "results",
) -> None:
    """client_factory(model_id) -> object with .generate(prompt) -> str,
    injectable for testing without loading a real model (see
    tests/test_run_pilot.py).
    """
    articles = load_sample(sample_csv)
    client = client_factory(model_id)
    model_dir = os.path.join(results_dir, model_id.replace("/", "__"))
    os.makedirs(model_dir, exist_ok=True)
    for article in articles:
        doi = article["doi"]
        out_path = os.path.join(model_dir, f"{doi.replace('/', '_')}.json")
        try:
            rows = extract_demographics_from_xml(doi, article["xml_path"], client)
            result = {"doi": doi, "model": model_id, "status": "ok", "rows": rows}
        except ExtractionValidationError as exc:
            result = {"doi": doi, "model": model_id, "status": "error", "error": str(exc)}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    print(f"{model_id}: wrote {len(articles)} results to {model_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default="data/sampled_articles.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "transformers"])
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--model", default=None, help="Override config.yaml's default_model")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_id = args.model or config["default_model"]

    def client_factory(mid: str):
        kwargs = {"tensor_parallel_size": args.tensor_parallel_size} if args.backend == "vllm" else {}
        return build_client(mid, backend=args.backend, **kwargs)

    run_pilot(args.sample, model_id, client_factory, args.results_dir)


if __name__ == "__main__":
    main()
