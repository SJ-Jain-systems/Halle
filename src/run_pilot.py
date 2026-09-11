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
import logging
import os

import yaml

from src.extract_demographics import ExtractionValidationError, extract_demographics_from_xml
from src.model_backend import build_client

logger = logging.getLogger(__name__)


def load_sample(csv_path: str) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _existing_status(out_path: str) -> str | None:
    """The `status` of a previously-written result file, or None if there is no
    readable result yet. Used to resume a preempted run."""
    if not os.path.exists(out_path):
        return None
    try:
        with open(out_path, encoding="utf-8") as f:
            return json.load(f).get("status")
    except (json.JSONDecodeError, OSError):
        return None


def run_pilot(
    sample_csv: str,
    model_id: str,
    client_factory,
    results_dir: str = "results",
    overwrite: bool = False,
) -> None:
    """client_factory(model_id) -> object with .generate(prompt) -> str,
    injectable for testing without loading a real model (see
    tests/test_run_pilot.py).

    Resumable and crash-proof, like src/run_pipeline.py: an article whose result
    file already says `status: "ok"` is skipped (unless `overwrite`), and any
    error on a single article is recorded as a `status: "error"` result rather
    than aborting the whole run. The model client is loaded lazily on the first
    article that actually needs generating, so a fully-resumed run never loads
    the model at all.
    """
    articles = load_sample(sample_csv)
    model_dir = os.path.join(results_dir, model_id.replace("/", "__"))
    os.makedirs(model_dir, exist_ok=True)

    client = None

    def get_client():
        nonlocal client
        if client is None:
            client = client_factory(model_id)
        return client

    counts = {"ok": 0, "error": 0, "skipped": 0}
    for article in articles:
        doi = article["doi"]
        out_path = os.path.join(model_dir, f"{doi.replace('/', '_')}.json")

        if not overwrite and _existing_status(out_path) == "ok":
            counts["skipped"] += 1
            continue

        try:
            rows = extract_demographics_from_xml(doi, article["xml_path"], get_client())
            result = {"doi": doi, "model": model_id, "status": "ok", "rows": rows}
            counts["ok"] += 1
        except ExtractionValidationError as exc:
            logger.warning("Extraction failed for %s: %s", doi, exc)
            result = {"doi": doi, "model": model_id, "status": "error", "error": str(exc)}
            raw = getattr(exc, "raw_output", None)
            if raw is not None:
                # Keep the raw generation (truncated) so parsing failures are
                # diagnosable without another GPU run.
                result["raw_output"] = raw[:4000]
            counts["error"] += 1
        except Exception as exc:  # keep going: one bad article must not kill the job
            logger.exception("Unexpected error extracting %s", doi)
            result = {"doi": doi, "model": model_id, "status": "error", "error": repr(exc)}
            counts["error"] += 1

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    logger.info(
        "%s: %d ok, %d error, %d skipped (of %d) -> %s",
        model_id, counts["ok"], counts["error"], counts["skipped"], len(articles), model_dir,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default="data/sampled_articles.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--backend", default="vllm", choices=["vllm", "transformers", "echo"],
        help="'echo' loads no model — a GPU-free dry-run to validate the wiring "
             "(CSV load, XML read, prompt build, JSON write) before requesting GPUs.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--model", default=None, help="Override config.yaml's default_model")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-run every article even if a prior 'ok' result exists (default: resume).",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_id = args.model or config["default_model"]

    def client_factory(mid: str):
        kwargs = {"tensor_parallel_size": args.tensor_parallel_size} if args.backend == "vllm" else {}
        return build_client(mid, backend=args.backend, **kwargs)

    run_pilot(args.sample, model_id, client_factory, args.results_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
