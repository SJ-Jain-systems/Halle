#!/bin/bash
# Full-scale demographic extraction over every article in
# data/corpus_index.csv, using the model chosen in config.yaml
# (meta-llama/Llama-3.3-70B-Instruct by default).
#
# Runs as a SLURM array job: each array task processes a disjoint shard of
# the index (--shard-index/--shard-count in src/run_pipeline.py) and appends
# to its own shard output file, so tasks never contend over one CSV. Combine
# shard outputs afterward (see docs/RUNNING_ON_RIVANNA.md step 5).
#
# Re-running a task (e.g. after preemption) is safe — src/run_pipeline.py
# skips DOIs already present in its shard's output file.
#
# Adjust --array=0-N to match SHARD_COUNT below (array index count must
# equal shard count, zero-indexed).
#
#   sbatch slurm/run_pipeline.slurm

#SBATCH --job-name=halle-pipeline
#SBATCH --account=<your_allocation>
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/run_pipeline_%A_%a.out

set -euo pipefail
mkdir -p logs data

module purge
module load miniforge
module load cuda
source activate halle

cd "$SLURM_SUBMIT_DIR"

SHARD_COUNT=8  # must match --array above (array goes 0..SHARD_COUNT-1)

python -m src.run_pipeline \
    --index data/corpus_index.csv \
    --out "data/demographics_table.shard${SLURM_ARRAY_TASK_ID}.csv" \
    --backend vllm \
    --tensor-parallel-size 4 \
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    --shard-count "$SHARD_COUNT"
