"""Run the 46-article pilot through our model (docs/DECISIONS.md #3,
meta-llama/Llama-3.3-70B-Instruct) and save the output so we can eyeball it
against docs/SCORING_RUBRIC.md before kicking off the full run.

Usage (on a Rivanna GPU node, see docs/RUNNING_ON_RIVANNA.md):
    python -m src.run_pilot --sample data/sampled_articles.csv --backend vllm

It reads the full text straight from the local corpus XML (the xml_path column
in data/sampled_articles.csv), so there are no network calls.
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
    """Return the status of a result file we already wrote, or None if there
    isn't one to read yet. This is what lets us pick back up after a job gets
    preempted."""
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
    """client_factory(model_id) hands back something with a .generate(prompt)
    method. We pass it in so tests can swap in a fake model instead of loading a
    real one (see tests/test_run_pilot.py).

    Like src/run_pipeline.py, this is safe to re-run. If an article's result file
    already says status "ok" we skip it (unless overwrite is set), and if one
    article blows up we write a status "error" result and keep going instead of
    taking down the whole run. The model client only gets built the first time we
    actually need to generate something, so a run that's fully resumed never
    loads the model at all.
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
            counts["error"] += 1
        except Exception as exc:  # one bad article shouldn't take down the whole run
            logger.exception("Unexpected error extracting %s", doi)
            result = {"doi": doi, "model": model_id, "status": "error", "error": repr(exc)}
            counts["error"] += 1

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    logger.info(
        "%s: %d ok, %d error, %d skipped out of %d, results in %s",
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
        help="'echo' loads no model. It's a GPU-free dry-run to check the wiring "
             "(read the CSV, read the XML, build the prompt, write the JSON) before you grab GPUs.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--model", default=None, help="Override config.yaml's default_model")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-run every article even if we already have an 'ok' result for it (by default we resume).",
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
