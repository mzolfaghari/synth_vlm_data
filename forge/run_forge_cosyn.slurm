#!/bin/bash
#SBATCH --job-name=forge_cosyn
#SBATCH --chdir=/home/reza/projects/nanoVLM
#SBATCH --output=logs/forge_cosyn/%j.out
#SBATCH --error=logs/forge_cosyn/%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --partition=VLM
#SBATCH --qos=normal

# FORGE over CoSyn QA, on our own Qwen (no external key): serve Qwen3.6-27B-FP8 (vision)
# DATA-PARALLEL x8 behind one OpenAI-compatible endpoint, then run forge_verify (deterministic
# gates -> adversarial panel on the residual) to produce a fidelity-checked manifest, and tear the
# server down. Mirrors chartgalaxy/run_full_8gpu.slurm.
#
# For the A/B pilot set MODE=single on one run and MODE=debate on another (different OUT), then
# compare kept-answer precision against a hand-checked reference.
#
# Inputs: a QA jsonl (rows {image, question, answer, [data], [tier]}) + the image dir.
# Override via env: QA_JSONL, IMAGE_DIR, OUT, MODE, LIMIT, ROUNDS, RMAX, WORKERS, MODEL, PORT, DP.
set -uo pipefail
export PYTHONUNBUFFERED=1
source ~/miniconda3/etc/profile.d/conda.sh
mkdir -p logs/forge_cosyn

MODEL="${MODEL:-Qwen/Qwen3.6-27B-FP8}"
PORT="${PORT:-18903}"
DP="${DP:-8}"
VLLM_ENV="${VLLM_ENV:-vlmeval}"
CLIENT_ENV="${CLIENT_ENV:-vlm-distill}"
WORKERS="${WORKERS:-32}"
MODE="${MODE:-debate}"                       # debate | single (single = A/B baseline)
ROUNDS="${ROUNDS:-2}"
RMAX="${RMAX:-1}"
LIMIT="${LIMIT:-0}"                          # 0 = all images
QA_JSONL="${QA_JSONL:-/home/reza/data/cosyn_qa.jsonl}"
IMAGE_DIR="${IMAGE_DIR:-/home/reza/data/cosyn_images}"
OUT="${OUT:-/home/reza/data/forge_cosyn_${MODE}.jsonl}"
CONFIG="${CONFIG:-forge_cosyn}"

[[ -f "$QA_JSONL" ]] || { echo "ERROR: QA_JSONL not found: $QA_JSONL"; exit 1; }
[[ -d "$IMAGE_DIR" ]] || { echo "ERROR: IMAGE_DIR not found: $IMAGE_DIR"; exit 1; }

# ── serve (background) ──
SRVLOG="logs/forge_cosyn/vllm_srv_${SLURM_JOB_ID}.log"
(
  conda activate "$VLLM_ENV"
  NVLIB=$(ls -d "$(conda info --base)/envs/${VLLM_ENV}"/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | grep -v '/cu13/' | tr '\n' ':')
  export LD_LIBRARY_PATH="${NVLIB}${LD_LIBRARY_PATH:-}"
  export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
  vllm serve "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --data-parallel-size "$DP" \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.88}" \
    --max-model-len 16384 \
    --reasoning-parser qwen3 \
    --max-num-seqs 256 \
    --no-async-scheduling --enforce-eager \
    --limit-mm-per-prompt '{"image":1}'
) > "$SRVLOG" 2>&1 &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null' EXIT

echo "waiting for endpoint http://127.0.0.1:${PORT} ..."
for i in $(seq 1 120); do
  curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && { echo "endpoint up after ~$((i*10))s"; break; }
  kill -0 $SRV_PID 2>/dev/null || { echo "SERVER DIED — see $SRVLOG"; tail -30 "$SRVLOG"; exit 1; }
  sleep 10
done
curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 || { echo "endpoint never came up"; tail -40 "$SRVLOG"; exit 1; }

# ── run FORGE (client) ──
conda activate "$CLIENT_ENV"
cd /home/reza/projects/nanoVLM/data_enrichment          # so `-m forge.forge_verify` resolves the package
QA_LLM_BASE_URL="http://127.0.0.1:${PORT}/v1" QA_LLM_MODEL="$MODEL" \
  python -m forge.forge_verify \
    --in "$QA_JSONL" --image-root "$IMAGE_DIR" --out "$OUT" \
    --config "$CONFIG" --mode "$MODE" --rounds "$ROUNDS" --rmax "$RMAX" \
    --source-key data --pass-source --workers "$WORKERS" --limit "$LIMIT" \
    --report "${OUT%.jsonl}_report.json"

echo "=== done (${MODE}) ==="
wc -l "$OUT"
