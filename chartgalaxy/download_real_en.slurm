#!/bin/bash
#SBATCH --job-name=cg_download
#SBATCH --chdir=/home/reza/projects/nanoVLM
#SBATCH --output=logs/cg_download/%j.out
#SBATCH --error=logs/cg_download/%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=main
#SBATCH --qos=normal

# STAGE 1 (CPU worker, network-bound): fetch real ChartGalaxy infographics, English-filtered,
# to /home/reza/data/chartgalaxy_real_en/. Resumable. No GPU.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CLIENT_ENV:-vlm-distill}"
mkdir -p logs/cg_download
srun python -u data_enrichment/chartgalaxy/download_real_en.py --limit "${LIMIT:-60000}"
echo "=== download done ==="
wc -l /home/reza/data/chartgalaxy_real_en/real_en_meta.jsonl
