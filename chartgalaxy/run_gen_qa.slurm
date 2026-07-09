#!/bin/bash
#SBATCH --job-name=cg_genqa
#SBATCH --chdir=/home/reza/projects/nanoVLM
#SBATCH --output=logs/cg_genqa/%j.out
#SBATCH --error=logs/cg_genqa/%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --partition=main
#SBATCH --qos=normal

# CPU CLIENT for ChartGalaxy QA generation — runs on a WORKER (never the login node).
# The vision-Qwen teacher must already be served (serve_qwen.slurm). Point at it:
#   QA_LLM_BASE_URL=http://<node>:18901/v1 LIMIT=20 sbatch data_enrichment/chartgalaxy/run_gen_qa.slurm
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CLIENT_ENV:-vlm-distill}"
mkdir -p logs/cg_genqa

: "${QA_LLM_BASE_URL:?set QA_LLM_BASE_URL=http://<serve-node>:18901/v1}"
export QA_LLM_MODEL="${QA_LLM_MODEL:-Qwen/Qwen3.6-27B-FP8}"

srun python -u data_enrichment/chartgalaxy/gen_qa.py \
  --base-url "${QA_LLM_BASE_URL}" \
  --num-questions "${NUMQ:-3}" \
  --workers "${WORKERS:-3}" \
  --limit "${LIMIT:-0}" \
  --out "${OUT:-/home/reza/data/chartgalaxy_real_en/real_en_qa.jsonl}"
echo "=== gen_qa client done ==="
