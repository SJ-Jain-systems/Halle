#!/bin/bash
# Final aggregation step: turns data/demographics_table.csv into the
# year-by-year / by-stage summaries that answer the brief's main research
# question. CPU-only, quick — safe to run on a login node interactively too,
# this script is provided mainly for a reproducible batch-logged run.
#
#   sbatch slurm/analyze_trends.slurm

#SBATCH --job-name=halle-analyze
#SBATCH --account=<your_allocation>
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/analyze_trends_%j.out

set -euo pipefail
mkdir -p logs data/analysis

module purge
module load miniforge

# See slurm/build_index.slurm for why this bypasses `conda activate`.
PYTHON_BIN="$HOME/.conda/envs/halle/bin/python"

cd "$SLURM_SUBMIT_DIR"
"$PYTHON_BIN" -m src.merge_shards --glob "data/demographics_table.shard*.csv" --out data/demographics_table.csv
"$PYTHON_BIN" -m src.analyze_trends --table data/demographics_table.csv --out-dir data/analysis
