# ChartGalaxy QA data-generation pipeline — full process

End-to-end pipeline that turns **real ChartGalaxy infographics** into an **InfoVQA-targeted SFT
dataset** of image→question→answer pairs, generated and verified by a vision LLM. This document is
the authoritative process reference; companion docs:
- [`qa_spec.md`](qa_spec.md) — question taxonomy (grounded in DVQA/PlotQA/ChartQA)
- [`SOTA_pipelines.md`](SOTA_pipelines.md) — survey of 12 SOTA chart-QA pipelines and what we adopted
- [`README.md`](README.md) — quick-start

**Rule:** everything runs on SLURM workers — never the login node (data-fetch, image work, and the
inference *client* all go through `sbatch`/`srun`).

---

## 1. Why this exists

Our model loses to LFM2.5 on **InfoVQA** (and TextVQA, MMVet). InfoVQA is *image-diversity bound*:
the ~4.4K unique InfographicVQA training images are exhausted, so more augmentation doesn't help —
it needs **genuinely new real infographics**. ChartGalaxy ships 61,833 real infographics (Statista,
Visual Capitalist, …) with underlying data tables but **no QA**. This pipeline generates the QA
ourselves — the same recipe the ChartGalaxy paper used for its +3.8–4.4 InfoVQA gain.

## 2. Architecture (two stages + a served VLM)

```
 real ChartGalaxy shards (HF)                Qwen3.6-27B-FP8 (vision), served DP×8
        │                                                  ▲
        ▼                                                  │ OpenAI-compatible /v1
  [Stage 1] download_real_en.py  ── images+meta ──▶ [Stage 2] gen_qa.py ──▶ real_en_qa.jsonl
   (CPU worker, langdetect gate)                    (worker client: generate → verify)
```

| Stage | Script | Resources | Output |
|---|---|---|---|
| 1. download + English filter | `download_real_en.py` (+ `download_real_en.slurm`) | 1 CPU node, network | `images/*.jpg` + `real_en_meta.jsonl` |
| serve the VLM | `benchmark_weakness_analysis/slurm/serve_qwen.slurm` (1 GPU) or `run_full_8gpu.slurm` (DP×8) | 1–8 GPU | OpenAI endpoint `:18901` |
| 2. generate + verify | `gen_qa.py` (+ `run_gen_qa.slurm`, or bundled in `run_full_8gpu.slurm`) | via endpoint | `real_en_qa.jsonl` |

Data lives under `/home/reza/data/chartgalaxy_real_en/`; code under this folder.

## 3. Stage 1 — fetch + English filter (`download_real_en.py`)

1. Stream the real ChartGalaxy shards (`real/dataset_*.tar`) from HuggingFace — each entry has a
   source **URL** + tabular `data` (images are **not** bundled).
2. **English pre-filter** — `langdetect` on the chart's title/label text; keep `en` prob > 0.90.
   (Cheap first pass; ~60–72% of real ChartGalaxy is English. It is *not* authoritative — charts
   whose labels are brand names/numbers can slip through; Stage 2 catches those from the image.)
3. Download the image from its URL (Statista CDN reliable; Pinterest often 403s), save as JPEG.
4. Emit one metadata record per kept image to `real_en_meta.jsonl`:
   `{id, config:"chartgalaxy_real_en", image (abs path), chart_type, source, lang, title, data}`.

Resumable (skips ids already in the meta).

## 4. Serving the VLM

`Qwen/Qwen3.6-27B-FP8` is vision-capable (`Qwen3_5ForConditionalGeneration`). Served via vLLM
(env `vlmeval`, vllm 0.22) as an OpenAI-compatible endpoint on port 18901, with the proven
anti-hang flags (`--enforce-eager`, `--no-async-scheduling`). For the full run we use
**data-parallel ×8** (`--data-parallel-size 8`) — the 27B fits on 1 GPU in FP8, so 8 replicas ≈ 8×
throughput behind one endpoint. **Thinking is disabled per-request** (`chat_template_kwargs:
{enable_thinking:false}`) — chart QA is perception, not multi-step reasoning; this raised smoke
yield from 5/20→18/20 images and removed timeouts.

## 5. Stage 2 — generate + verify (`gen_qa.py`), the core

For each English infographic, **two VLM calls in image mode**:

**(a) Generation** — the VLM sees the **image + table + chart_type** and returns up to 3 QA pairs as
JSON, each tagged with a `tier` from the grounded taxonomy and a one-sentence `rationale`:
- `structural` (count series/bars, element identity, legend), `data_retrieval` (single-value
  lookup), `reasoning` (extremum, comparison yes/no, aggregation, difference/ratio, trend),
  `visual` (color/icon/position). Answers constrained to a **single number or word**.

**(b) Verification** (image-based, one call per pair) — the VLM is re-shown the image with
(question, proposed answer) and returns `{english, correct, corrected_answer}`:
- `english:false` → the whole image is dropped (catches non-English charts langdetect missed, e.g.
  a French Statista chart whose labels were English brand names).
- `correct:true` → keep. Else keep `corrected_answer` if it passes the answer guard.

### Quality controls (what makes the labels trustworthy)

| Control | Rule | Grounded in |
|---|---|---|
| **Single-value answers** | reject > 8 words or ≥ 2 commas; applied to generated **and** corrected answers | ChartQA/InfoVQA answer format |
| **Image English gate** | verify call reports `english`; non-English image dropped | our fix for langdetect gap |
| **Image-based correctness** | every pair re-checked against the rendered chart | ChartGalaxy method |
| **Numeric groundedness ±5%** | for `data_retrieval`/`structural` numeric answers, the value must appear in the table within 5% (`grounded_numeric`) — drops fabricated lookups | ChartQA "answer must be in table" + ChartX ±5% |
| **Dedup** | reject a question with ROUGE-L ≥ 0.7 vs an already-kept question for that image | Self-Instruct |
| **Tier balance** | ≤ 2 kept pairs per tier per image | CharXiv descriptive/reasoning split |
| **CoT rationale** | one-sentence `rationale` stored per pair (esp. reasoning) | ChartAssistant (+20% acc), CoSyn |

Output row: `{config, image, question, answer, tier, rationale}` → `real_en_qa.jsonl` (resumable;
`--out` selects a fresh path for a v2 re-run).

## 6. Results

### v2 (current, recommended) — `real_en_qa_v2.jsonl`
Full run with all §5 quality upgrades (rationale + numeric groundedness ±5% + dedup + tier balance):
- **80,951** verified pairs / **31,338** English infographics (**2.58/image**)
- **Rationale on 100%** of pairs; answers single-value (max 8 words, median 1); **0 verbose** — no
  post-cleaning needed (the corrected-answer guard applies to the verifier output too)
- Tier mix: reasoning ~51% · data_retrieval ~34% · structural ~8% · visual ~7%
- Schema: `{config, image, question, answer, tier, rationale}`
- Generation ~3h15m on 8 GPU DP×8 (initial run + one resume — see note)

> **Infra note:** the DP×8 vLLM server is prone to a mid-run replica crash (known Qwen-VL+vLLM
> multimodal instability). The first pass died ~2/3 through (frozen at 27,521 pairs); because
> `gen_qa` is resumable (`--out` skips done images), a second submit at `GPU_MEM_UTIL=0.88`
> finished the rest with 0 errors. **If a full run ends with far fewer productive images than
> expected, just re-submit the same job — it resumes.**

### v1 (pre-upgrade, fallback) — `real_en_qa_clean.jsonl`
First run, before the upgrades: **92,670** pairs / **31,555** images (2.94/img), single-value
answers but **no rationale/groundedness/dedup**. Kept as a fallback; **v2 supersedes it.**

## 7. Reproduce / regenerate (v2, with all upgrades)

```bash
# Stage 1 — fetch English real infographics (CPU worker)
sbatch data_enrichment/chartgalaxy/download_real_en.slurm            # LIMIT=60000 default

# Stages "serve + generate" bundled on one 8-GPU node, writing a fresh v2 output:
#   (edit run_full_8gpu.slurm's gen_qa call to add: --out .../real_en_qa_v2.jsonl)
sbatch data_enrichment/chartgalaxy/run_full_8gpu.slurm

# — or — serve + client separately:
sbatch benchmark_weakness_analysis/slurm/serve_qwen.slurm            # prints <node>:18901
QA_LLM_BASE_URL=http://<node>:18901/v1 LIMIT=20 \
  sbatch data_enrichment/chartgalaxy/run_gen_qa.slurm               # 20-image smoke first
```

Recommended: run a **20-image smoke** (`LIMIT=20`, `--out` a temp file) and eyeball
`tier`/`rationale`/groundedness before the full regeneration.

## 8. Scope & honest limits

- **Synthetic shards not yet processed** — this covers the ~35K English *real* infographics. The
  1.7M *synthetic* shards (bundled PNGs, all English) are a separate, larger run using the same
  `gen_qa.py` (point Stage 1 at the synthetic images instead).
- **Self-verification** — same model generates and verifies. The image re-check + numeric
  groundedness mitigate this; we deliberately skip k-vote/2nd-model (SOTA relies on table-grounding,
  not multi-vote — see `SOTA_pipelines.md`).
- **Not implemented** (low priority): PoT-execution round-trip (TinyChart), unanswerable-abstention
  (ChartQAPro). Rationale for skipping is in `SOTA_pipelines.md`.
