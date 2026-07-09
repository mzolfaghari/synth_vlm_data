# COMPACT — compositional atomic-to-complex QA generation

Integration of the **COMPACT** data recipe (COMPositional Atomic-to-complex visual Capability
Tuning) into this repo. COMPACT scales the *compositional complexity* of each training example:
for an image it samples `k` atomic visual capabilities and generates one question that integrates
**exactly** those `k`, verifies it, and assembles LLaVA-style conversation data. In the paper this
lets ~10% of LLaVA-665K reach ~100% of full-data performance across 8 benchmarks.

> **Upstream / attribution.** Method and code from
> **COMPACT** — Yang et al., *"COMPACT: COMPositional Atomic-to-complex Visual Capability Tuning"*,
> arXiv:[2504.21850](https://arxiv.org/abs/2504.21850) · repo
> [github.com/princetonvisualai/compact](https://github.com/princetonvisualai/compact).
> `config.py`, `prompts.py`, `utils.py`, `generator.py`, `verifier.py`, and the core of
> `processor.py`/`main.py` are **copied from that repository** (which carries **no LICENSE file** —
> treat as research use; clear terms with the authors before commercial use). Our additions are
> `backends.py`, `to_manifest.py`, and `run_compact_cosyn_8gpu.slurm`.

## Atomic capability taxonomy (10)

`spatial_relationship`, `object_interaction`, `text_recognition`, `spatial_recognition`,
`action_recognition`, `object_recognition`, `counting`, `color`, `shape`, `scene_understanding`
(definitions live in `prompts.py`).

## What changed vs. upstream

Upstream calls **Gemini 2.0 Flash** via `google.generativeai`. We keep that path, but the default
backend is **our self-hosted Qwen (vLLM, OpenAI-compatible)** so it runs on our cluster with no
external API key. This is done non-invasively: `backends.py` provides `OpenAICompatClient`, a
drop-in that mimics the `genai` interface the upstream `generator.py`/`verifier.py` expect
(`client.GenerativeModel(name).generate_content(contents=[...], generation_config={...}).text`).
Only `processor.py`/`main.py` were touched (to build the client via `make_client`).

## Files

| File | Origin | Purpose |
|---|---|---|
| `config.py`, `prompts.py`, `utils.py` | upstream (verbatim) | taxonomy, generation/verification prompts, JSON cleanup |
| `generator.py`, `verifier.py` | upstream (verbatim) | per-image generation; capability verification (`!=k` → reject) |
| `processor.py`, `main.py` | upstream + minimal edits | orchestration; CLI + backend selection |
| `backends.py` | **ours** | Qwen-vLLM (OpenAI) / Gemini client behind the `genai` interface |
| `to_manifest.py` | **ours** | COMPACT output → `{config,image,question,answer}` training manifest |
| `run_compact_cosyn_8gpu.slurm` | **ours** | serve Qwen ×8 + generate k=1,2,3 over CoSyn images + assemble manifest |

## Run

### On our Qwen (default) over CoSyn — one command

```bash
# 1) get CoSyn images on disk (see ../cosyn/download.py) -> IMAGE_DIR
# 2) serve Qwen ×8 + generate k=1,2,3 + assemble a manifest, all on a worker:
IMAGE_DIR=/home/reza/data/cosyn_images NUM_SAMPLES=20000 \
  sbatch data_enrichment/compact/run_compact_cosyn_8gpu.slurm
# -> /home/reza/data/compact_cosyn/compact_cosyn.jsonl  ({config,image,question,answer})
```

`k` is fixed per `main.py` run (as upstream); the paper's k∈{1,2,3} mix comes from running each `k`
and merging — the runner does all three and `to_manifest.py` merges them.

### Manual / any image dir

```bash
cd data_enrichment
# against our served Qwen:
QA_LLM_BASE_URL=http://<node>:<port>/v1 QA_LLM_MODEL=Qwen/Qwen3.6-27B-FP8 \
  python -m compact.main --backend openai --k 3 --image_dir /path/imgs --output_dir out --processes 64

# or faithfully with Gemini (upstream behaviour):
python -m compact.main --backend gemini --api_key "$GEMINI_API_KEY" --k 3 \
  --image_dir /path/imgs --output_dir out --processes 32

# then convert COMPACT output -> our manifest:
python -m compact.to_manifest --inputs out/output_k1.json out/output_k2.json out/output_k3.json \
  --image-root /path/imgs --config compact_cosyn --out /home/reza/data/compact_cosyn.jsonl
```

The resulting manifest has absolute image paths, so it folds into `finetune_mixed.py --mixed`
alongside the chartgalaxy / cosyn manifests.
