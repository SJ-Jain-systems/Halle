# ============================================================================
# PLAIN-ENGLISH NOTES (for colleagues reading this file)
#
# What this file is for: the big one. This runs the model over ALL 58,000 papers
# and writes the final demographics spreadsheet (one row per participant sample).
# Everything before this was preparation; this is the step that produces the
# data the study actually analyzes.
#
# Two features make it survivable on a shared supercomputer:
#   - SHARDING: the work is split into equal chunks ("shards"). We launch many
#     copies at once, each handling every Nth paper, so 58,000 papers get done
#     in parallel across many GPUs instead of one slow line.
#   - RESTARTABLE: before working on a paper it checks whether that paper is
#     already in the output file, and skips it if so. So if a job gets killed
#     halfway (which happens on shared clusters), you just re-run it and it
#     picks up where it stopped instead of starting over.
#
# If a single paper fails (bad file, weird model output), it logs the problem
# and moves on rather than crashing the whole multi-hour run.
# ============================================================================

"""Full-scale demographic extraction over the entire filtered corpus
(brief items 1-2: this is what actually answers the research question, as
opposed to src/run_pilot.py which only covers the 10-article pilot).

Reads data/corpus_index.csv (src/build_corpus_index.py), runs every article
through the single chosen model (docs/DECISIONS.md #3,
meta-llama/Llama-3.3-70B-Instruct by default, from config.yaml), and appends
one row per extracted sample to an output CSV.

Designed to run as a SLURM array job (slurm/run_pipeline.slurm): pass
--shard-index/--shard-count to have each array task process a disjoint slice
of the index. Safe to re-run — articles already present in the output CSV
are skipped, so a killed/preempted job just picks up where it left off.

Usage:
    python -m src.run_pipeline --index data/corpus_index.csv \\
        --out data/demographics_table.csv --backend vllm --tensor-parallel-size 4
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

# The columns of the final demographics table. Demographic fields come from the
# model; the metadata fields (subfield, year, stage, ...) are copied from the
# master list so each row is self-contained for analysis.
OUTPUT_FIELDS = [
    "doi",
    "sample_id",
    "subfield",
    "matched_subfields",
    "subject_level_1",
    "subject",
    "publication_date",
    "year",
    "stage",
    "lead_institution",
    "gender_reported",
    "gender_pct",
    "race_reported",
    "race_pct",
    "education_reported",
    "education_pct",
    "ses_reported",
    "ses_value",
]


def load_index(index_csv: str) -> list[dict]:
    with open(index_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def already_processed_dois(out_path: str) -> set[str]:
    # Read whatever is already in the output file so we can skip those papers.
    # This is what makes the run restartable after an interruption.
    if not os.path.exists(out_path):
        return set()
    with open(out_path, newline="", encoding="utf-8") as f:
        return {row["doi"] for row in csv.DictReader(f)}


def shard(rows: list[dict], shard_index: int, shard_count: int) -> list[dict]:
    # Split the work: shard 3 of 8 handles rows 3, 11, 19, 27, ... This "every
    # Nth row" trick means each parallel job gets a fair, non-overlapping slice.
    if shard_count <= 1:
        return rows
    return [row for i, row in enumerate(rows) if i % shard_count == shard_index]


def run(
    index_csv: str,
    out_path: str,
    client,
    shard_index: int = 0,
    shard_count: int = 1,
) -> None:
    rows = load_index(index_csv)
    rows = shard(rows, shard_index, shard_count)   # take only this job's slice
    done = already_processed_dois(out_path)        # skip anything already done

    write_header = not os.path.exists(out_path)
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        for i, article in enumerate(rows):
            doi = article["doi"]
            if doi in done:
                continue
            # Run the model on this paper. If anything goes wrong - bad output,
            # unreadable file - log it and move on. One bad paper must not sink
            # an hours-long run.
            try:
                samples = extract_demographics_from_xml(doi, article["xml_path"], client)
            except ExtractionValidationError as exc:
                logger.warning("Extraction failed for %s: %s", doi, exc)
                continue
            except Exception:
                logger.exception("Unexpected error extracting %s", doi)
                continue

            for sample in samples:
                # Serialize dict-valued fields to JSON text so they round-trip
                # cleanly through CSV (csv.writer would otherwise fall back to
                # Python repr(), which json.loads() can't parse back).
                sample = {
                    **sample,
                    "gender_pct": json.dumps(sample.get("gender_pct") or {}),
                    "race_pct": json.dumps(sample.get("race_pct") or {}),
                    "education_pct": json.dumps(sample.get("education_pct") or {}),
                }
                # Write the model's answer plus the metadata copied from the
                # master list, so each row stands on its own in the analysis.
                writer.writerow(
                    {
                        **sample,
                        "subfield": article.get("matched_subfields", "").split(";")[0],
                        "matched_subfields": article.get("matched_subfields", ""),
                        "subject_level_1": article.get("subject_level_1", ""),
                        "subject": article.get("subject", ""),
                        "publication_date": article.get("publication_date", ""),
                        "year": article.get("year", ""),
                        "stage": article.get("stage", ""),
                        "lead_institution": article.get("lead_institution", ""),
                    }
                )
            f.flush()  # write to disk as we go, so progress survives a crash
            if (i + 1) % 50 == 0:
                logger.info("Processed %d/%d articles in this shard", i + 1, len(rows))

    logger.info("Shard %d/%d done: %d articles in %s", shard_index, shard_count, len(rows), index_csv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="data/corpus_index.csv")
    parser.add_argument("--out", default="data/demographics_table.csv")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "transformers"])
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--model", default=None, help="Override config.yaml's default_model")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()

    # The model name lives in config.yaml (Llama-3.3-70B by default) unless
    # overridden on the command line.
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_id = args.model or config["default_model"]

    kwargs = {"tensor_parallel_size": args.tensor_parallel_size} if args.backend == "vllm" else {}
    client = build_client(model_id, backend=args.backend, **kwargs)

    run(args.index, args.out, client, shard_index=args.shard_index, shard_count=args.shard_count)


if __name__ == "__main__":
    main()
