#!/bin/bash
#SBATCH --job-name=compact_cosyn
#SBATCH --chdir=/home/reza/projects/nanoVLM
#SBATCH --output=logs/compact_cosyn/%j.out
#SBATCH --error=logs/compact_cosyn/%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --partition=VLM
#SBATCH --qos=normal

# Run the COMPACT compositional-QA pipeline over CoSyn images, on our own Qwen (no Gemini key):
# serve Qwen3.6-27B-FP8 (vision) DATA-PARALLEL x8 behind one OpenAI-compatible endpoint, then run
# compact/main.py for k=1,2,3 (the paper's compositional mix) with --backend openai against that
# endpoint, merge the three into one {config,image,question,answer} manifest, and tear the server
# down. Mirrors chartgalaxy/run_full_8gpu.slurm.
#
# Prereq: CoSyn images on disk (data_enrichment/cosyn/download.py -> IMAGE_DIR).
# Override via env: IMAGE_DIR, OUT_DIR, MANIFEST, NUM_SAMPLES, CONFIG, KS, PROCESSES, MODEL, PORT, DP.
set -uo pipefail
export PYTHONUNBUFFERED=1
source ~/miniconda3/etc/profile.d/conda.sh
mkdir -p logs/compact_cosyn

MODEL="${MODEL:-Qwen/Qwen3.6-27B-FP8}"
PORT="${PORT:-18902}"
DP="${DP:-8}"
VLLM_ENV="${VLLM_ENV:-vlmeval}"
CLIENT_ENV="${CLIENT_ENV:-vlm-distill}"
PROCESSES="${PROCESSES:-64}"
KS="${KS:-1 2 3}"                                   # compositional complexity levels to generate
NUM_SAMPLES="${NUM_SAMPLES:-20000}"                 # images sampled per k
IMAGE_DIR="${IMAGE_DIR:-/home/reza/data/cosyn_images}"
OUT_DIR="${OUT_DIR:-/home/reza/data/compact_cosyn}"
CONFIG="${CONFIG:-compact_cosyn}"
MANIFEST="${MANIFEST:-${OUT_DIR}/compact_cosyn.jsonl}"

[[ -d "$IMAGE_DIR" ]] || { echo "ERROR: IMAGE_DIR not found: $IMAGE_DIR (build via cosyn/download.py)"; exit 1; }
mkdir -p "$OUT_DIR"

# ── serve (background) ──
SRVLOG="logs/compact_cosyn/vllm_srv_${SLURM_JOB_ID}.log"
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

# ── wait for readiness ──
echo "waiting for endpoint http://127.0.0.1:${PORT} ..."
for i in $(seq 1 120); do
  curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && { echo "endpoint up after ~$((i*10))s"; break; }
  kill -0 $SRV_PID 2>/dev/null || { echo "SERVER DIED — see $SRVLOG"; tail -30 "$SRVLOG"; exit 1; }
  sleep 10
done
curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 || { echo "endpoint never came up"; tail -40 "$SRVLOG"; exit 1; }

# ── generate (client): run COMPACT for each k against our endpoint ──
conda activate "$CLIENT_ENV"
cd /home/reza/projects/nanoVLM/data_enrichment           # so `-m compact.main` resolves the package
export QA_LLM_BASE_URL="http://127.0.0.1:${PORT}/v1" QA_LLM_MODEL="$MODEL"
INPUTS=()
for k in ${KS}; do
  echo "=== COMPACT k=${k} over ${IMAGE_DIR} (${NUM_SAMPLES} imgs, ${PROCESSES} procs) ==="
  python -m compact.main --backend openai --k "${k}" \
    --image_dir "${IMAGE_DIR}" --output_dir "${OUT_DIR}" \
    --num_samples "${NUM_SAMPLES}" --processes "${PROCESSES}"
  INPUTS+=("${OUT_DIR}/output_k${k}.json")
done

# ── assemble one training manifest {config,image,question,answer} (absolute image paths) ──
python -m compact.to_manifest --inputs "${INPUTS[@]}" \
  --image-root "${IMAGE_DIR}" --config "${CONFIG}" --out "${MANIFEST}"

echo "=== done ==="
wc -l "${MANIFEST}"
