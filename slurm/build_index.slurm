#!/bin/bash
# Scan the local allofplos corpus and write data/corpus_index.csv.
# CPU-only, single node — no GPU needed for this step.
#
# Adjust --account/--partition to match your Rivanna allocation before
# submitting (see docs/RUNNING_ON_RIVANNA.md). Submit with:
#   sbatch slurm/build_index.slurm

#SBATCH --job-name=halle-build-index
#SBATCH --account=<your_allocation>
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/build_index_%j.out

set -euo pipefail
mkdir -p logs data

module purge
module load miniforge  # or `anaconda`/`python` — check `module avail` on your Rivanna instance

# `conda activate` was unreliable at rewriting PATH in this environment (both
# in batch jobs and interactively) — verified live that it left `python`
# resolving to the base miniforge install instead of the `halle` env even
# though `conda env list` reported `halle` as active. Calling the env's
# interpreter by absolute path sidesteps that entirely.
PYTHON_BIN="$HOME/.conda/envs/halle/bin/python"

cd "$SLURM_SUBMIT_DIR"
"$PYTHON_BIN" -m src.build_corpus_index \
    --corpus-dir "${PLOS_CORPUS:-$HOME/allofplos_corpus}" \
    --out data/corpus_index.csv
