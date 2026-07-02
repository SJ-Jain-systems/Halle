#!/bin/bash
# Run the 10-article pilot through meta-llama/Llama-3.3-70B-Instruct and
# write results/meta-llama__Llama-3.3-70B-Instruct/<doi>.json for scoring
# against docs/SCORING_RUBRIC.md.
#
# Kept in its own job (rather than looping over both candidate models in one
# process) so GPU memory from one 70B/7B load never has to be released
# before the next model loads — submit slurm/pilot_comparison_mistral.slurm
# separately for the other model.
#
#   sbatch slurm/pilot_comparison_llama.slurm

#SBATCH --job-name=halle-pilot-llama
#SBATCH --account=<your_allocation>
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pilot_llama_%j.out

set -euo pipefail
mkdir -p logs results

module purge
module load miniforge
module load cuda   # match the CUDA version your torch/vllm build expects
source activate halle

cd "$SLURM_SUBMIT_DIR"
python -m src.compare_llms \
    --sample data/sampled_articles.csv \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --backend vllm \
    --tensor-parallel-size 4
