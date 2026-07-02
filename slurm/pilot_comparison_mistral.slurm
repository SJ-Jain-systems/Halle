#!/bin/bash
# Run the 10-article pilot through mistralai/Mistral-7B-Instruct-v0.3 and
# write results/mistralai__Mistral-7B-Instruct-v0.3/<doi>.json for scoring
# against docs/SCORING_RUBRIC.md. See pilot_comparison_llama.slurm for why
# this is a separate job rather than one job looping over both models.
#
#   sbatch slurm/pilot_comparison_mistral.slurm

#SBATCH --job-name=halle-pilot-mistral
#SBATCH --account=<your_allocation>
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pilot_mistral_%j.out

set -euo pipefail
mkdir -p logs results

module purge
module load miniforge
module load cuda
source activate halle

cd "$SLURM_SUBMIT_DIR"
python -m src.compare_llms \
    --sample data/sampled_articles.csv \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --backend vllm \
    --tensor-parallel-size 1
