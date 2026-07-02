"""Run the 10-article pilot through each candidate model and save raw +
parsed output for scoring against docs/SCORING_RUBRIC.md.

Usage:
    python -m src.compare_llms --sample data/sampled_articles.csv

Requires:
  - data/sampled_articles.csv (produced by src/sample_articles.py)
  - a ModelClient implementation wired to real inference (Hugging Face
    Inference API, a local transformers pipeline, etc.) — this sandbox has
    neither network access to huggingface.co nor API credentials, so this
    harness has not been run end-to-end. See docs/DECISIONS.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import yaml

from src.extract_demographics import (
    ExtractionValidationError,
    ModelClient,
    extract_demographics,
)
from src.plos_client import fetch_fulltext


def load_sample(csv_path: str) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_comparison(sample_csv: str, models: list[str], client_factory, results_dir: str = "results") -> None:
    """client_factory(model_id) -> ModelClient, injectable for testing."""
    articles = load_sample(sample_csv)
    for model_id in models:
        client = client_factory(model_id)
        model_dir = os.path.join(results_dir, model_id.replace("/", "__"))
        os.makedirs(model_dir, exist_ok=True)
        for article in articles:
            doi = article["doi"]
            article_text = fetch_fulltext(doi)
            out_path = os.path.join(model_dir, f"{doi.replace('/', '_')}.json")
            try:
                rows = extract_demographics(doi, article_text, client)
                result = {"doi": doi, "model": model_id, "status": "ok", "rows": rows}
            except ExtractionValidationError as exc:
                result = {"doi": doi, "model": model_id, "status": "error", "error": str(exc)}
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default="data/sampled_articles.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    models = config["candidate_models"]

    def client_factory(model_id: str) -> ModelClient:
        return ModelClient(model_id=model_id)  # replace with a real backend before running

    run_comparison(args.sample, models, client_factory, args.results_dir)


if __name__ == "__main__":
    main()
