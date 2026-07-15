#!/bin/bash
# Run the 46-article pilot through meta-llama/Llama-3.3-70B-Instruct (the model
# we picked in docs/DECISIONS.md #3) and write
# results/meta-llama__Llama-3.3-70B-Instruct/<doi>.json. It's a QA spot-check
# against docs/SCORING_RUBRIC.md before we spend GPU time on the full corpus run.
#
#   sbatch slurm/run_pilot.slurm

#SBATCH --job-name=halle-pilot
#SBATCH --account=<your_allocation>
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/run_pilot_%j.out

set -euo pipefail
mkdir -p logs results

module purge
module load miniforge
module load cuda   # match the CUDA version your torch/vllm build expects

# See slurm/build_index.slurm for why this bypasses `conda activate`.
PYTHON_BIN="$HOME/.conda/envs/halle/bin/python"

cd "$SLURM_SUBMIT_DIR"
"$PYTHON_BIN" -m src.run_pilot \
    --sample data/sampled_articles.csv \
    --backend vllm \
    --tensor-parallel-size 4
