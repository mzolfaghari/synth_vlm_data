#!/bin/bash
#SBATCH --job-name=cg_full_8gpu
#SBATCH --chdir=/home/reza/projects/nanoVLM
#SBATCH --output=logs/cg_full/%j.out
#SBATCH --error=logs/cg_full/%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --partition=VLM
#SBATCH --qos=normal

# STAGE 2 (1 node, 8 GPU): serve Qwen3.6-27B-FP8 (vision) DATA-PARALLEL x8 behind one endpoint,
# then run the ChartGalaxy gen+verify client at high concurrency over the downloaded English
# images, then tear the server down. Generation & verification are BOTH Qwen (thinking disabled).
# Data -> /home/reza/data/chartgalaxy_real_en/real_en_qa.jsonl (resumable).
set -uo pipefail
export PYTHONUNBUFFERED=1
source ~/miniconda3/etc/profile.d/conda.sh
mkdir -p logs/cg_full

MODEL="${MODEL:-Qwen/Qwen3.6-27B-FP8}"
PORT="${PORT:-18901}"
DP="${DP:-8}"                       # 8 data-parallel replicas, 1 GPU each (27B FP8 fits on 1)
VLLM_ENV="${VLLM_ENV:-vlmeval}"     # vllm 0.22 + transformers for qwen3_5
CLIENT_ENV="${CLIENT_ENV:-vlm-distill}"
WORKERS="${WORKERS:-96}"

# ── serve (background) — flags ported from benchmark_weakness_analysis/slurm/serve_qwen.slurm ──
SRVLOG="logs/cg_full/vllm_srv_${SLURM_JOB_ID}.log"
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

# ── wait for readiness (up to ~20 min for 8 replicas to load) ──
echo "waiting for endpoint http://127.0.0.1:${PORT} ..."
for i in $(seq 1 120); do
  curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && { echo "endpoint up after ~$((i*10))s"; break; }
  kill -0 $SRV_PID 2>/dev/null || { echo "SERVER DIED — see $SRVLOG"; tail -30 "$SRVLOG"; exit 1; }
  sleep 10
done
curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 || { echo "endpoint never came up"; tail -40 "$SRVLOG"; exit 1; }

# ── generate + verify (client) at high concurrency over ALL downloaded English images ──
conda activate "$CLIENT_ENV"
QA_LLM_BASE_URL="http://127.0.0.1:${PORT}/v1" QA_LLM_MODEL="$MODEL" \
  python -u data_enrichment/chartgalaxy/gen_qa.py --num-questions 3 --workers "$WORKERS" \
  --meta "${META:-/home/reza/data/chartgalaxy_real_en/real_en_meta.jsonl}" \
  --out "${OUT:-/home/reza/data/chartgalaxy_real_en/real_en_qa.jsonl}"

echo "=== full run done ==="
wc -l /home/reza/data/chartgalaxy_real_en/real_en_qa.jsonl
