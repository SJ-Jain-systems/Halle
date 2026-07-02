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
# `conda activate` needs its shell function sourced explicitly here — a batch
# script runs non-interactively, so the usual .bashrc-triggered conda init
# never fires and a bare `conda activate` silently no-ops (no error, wrong env).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate halle   # conda env created per docs/RUNNING_ON_RIVANNA.md step 0

cd "$SLURM_SUBMIT_DIR"
python -m src.build_corpus_index \
    --corpus-dir "${PLOS_CORPUS:-$HOME/allofplos_corpus}" \
    --out data/corpus_index.csv
