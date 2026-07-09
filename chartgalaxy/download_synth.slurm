#!/bin/bash
#SBATCH --job-name=cg_synth_dl
#SBATCH --chdir=/home/reza/projects/nanoVLM
#SBATCH --output=logs/cg_download/%j.out
#SBATCH --error=logs/cg_download/%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --partition=main
#SBATCH --qos=normal

# STAGE 1 (SYNTHETIC, CPU worker): extract bundled synthetic ChartGalaxy infographics
# to /home/reza/data/chartgalaxy_synth/. Resumable. No GPU.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CLIENT_ENV:-vlm-distill}"
mkdir -p logs/cg_download
srun python -u data_enrichment/chartgalaxy/download_synth.py --limit "${LIMIT:-200000}"
echo "=== synth download done ==="
wc -l /home/reza/data/chartgalaxy_synth/synth_meta.jsonl
