"""Merge the per-shard CSVs written by a slurm/run_pipeline.slurm array job
into the single data/demographics_table.csv that src/analyze_trends.py reads.

Usage:
    python -m src.merge_shards --glob "data/demographics_table.shard*.csv" \\
        --out data/demographics_table.csv
"""
from __future__ import annotations

import argparse
import csv
import glob


def merge(shard_glob: str, out_path: str) -> int:
    shard_paths = sorted(glob.glob(shard_glob))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files matched {shard_glob!r}")

    total_rows = 0
    fieldnames = None
    with open(out_path, "w", newline="", encoding="utf-8") as out_f:
        writer = None
        for shard_path in shard_paths:
            with open(shard_path, newline="", encoding="utf-8") as in_f:
                reader = csv.DictReader(in_f)
                if writer is None:
                    fieldnames = reader.fieldnames
                    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)
                    total_rows += 1
    return total_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", default="data/demographics_table.shard*.csv")
    parser.add_argument("--out", default="data/demographics_table.csv")
    args = parser.parse_args()
    total = merge(args.glob, args.out)
    print(f"Merged {total} rows into {args.out}")


if __name__ == "__main__":
    main()
