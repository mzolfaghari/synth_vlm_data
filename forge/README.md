# FORGE — grounded compositional VQA synthesis, tempered by adversarial adjudication

FORGE turns a QA set over programmatically **rendered** images (charts / docs / tables — e.g.
CoSyn) into a **fidelity-checked** training manifest. Its distinctive stage is an *adversarial
panel*: a rigid Advocate defends each answer while a **lenient** judge (accepts if roughly right)
and a **strict** judge (accepts only if exact) independently decide by looking at the image, and a
pair is kept only when **both agree**. Cheap sanity gates run first, but only to drop junk — every
surviving pair is verified by the panel.

![pipeline](docs/forge_pipeline.png)  <!-- source: docs/forge_pipeline.tex · vector: docs/forge_pipeline.pdf -->

## Why (and where the idea comes from)

Our earlier generators verify an answer with a **single** model pass, which tends to rubber-stamp
its own mistakes. The adversarial-debate verification here is adapted from the asymmetric-debate
method of Mazza & Levi, *"Synthetic Training of Custom Policy Guardrails via Asymmetric Debate"*
([arXiv:2604.25203](https://arxiv.org/abs/2604.25203)) — whose ablation shows debate verification
far exceeds both no-verification and single-model self-refinement. We retarget it from text policy
classification to **VQA answer-correctness**, and add a cheap deterministic first filter so it is
affordable at scale. (Only the method is reused; no upstream code — that repo ships the benchmark,
not the generator.)

## Pipeline (see the figure)

1. **Input** — rendered images that ship with their structured source (data table / render code) →
   free ground truth for both generation and judging.
2. **Compose** *(optional, upstream)* — a compositional generator proposes `Q + candidate A` by
   sampling `k` skills **weighted by the model's weakness profile** (from benchmark weakness
   analysis), so we oversample the skills the model is worst at (e.g. `../compact`). FORGE can
   also just verify an existing QA set.
3. **Gates** (`gates.py`, no LLM) — cheap sanity filters that only **drop junk** (not a single-value
   answer / a near-duplicate). They do **not** judge correctness — every survivor goes to the panel.
4. **Adjudicate** (`debate.py`) — a rigid Advocate + a **lenient** judge (accepts if roughly right)
   and a **strict** judge (accepts only if exact) debate up to `T` rounds; **both agree → accept**;
   **disagree → refine `A` and re-adjudicate ≤ `R_max`, else drop**.
5. **Assemble** — `{config, image, question, answer}` (+ `tier`), ready for `finetune_mixed.py --mixed`.

## Files

| File | Purpose |
|---|---|
| `gates.py` | cheap sanity filters + `triage()` (drop junk / send to panel) |
| `prompts.py` | advocate + lenient/strict judge + single-judge prompts (VQA correctness) |
| `debate.py` | `adjudicate()` (asymmetric debate + refinement) and `single_verify()` (baseline) |
| `forge_verify.py` | orchestrator CLI: gates drop junk → panel verifies every survivor, per image, resumable |
| `llm.py` | OpenAI-compatible chat helper for our Qwen vLLM endpoint |
| `run_forge_cosyn.slurm` | serve Qwen ×8 + run FORGE over CoSyn QA on a worker |
| `docs/forge_pipeline.tex` | the figure above (TikZ source) |

## Run

```bash
# serve Qwen x8 + gates + adversarial panel over CoSyn QA, on a worker:
QA_JSONL=/home/reza/data/cosyn_qa.jsonl IMAGE_DIR=/home/reza/data/cosyn_images MODE=debate \
  sbatch data_enrichment/forge/run_forge_cosyn.slurm
# -> /home/reza/data/forge_cosyn_debate.jsonl  ({config,image,question,answer})
```

Manual / any endpoint:

```bash
cd data_enrichment
QA_LLM_BASE_URL=http://<node>:<port>/v1 QA_LLM_MODEL=Qwen/Qwen3.6-27B-FP8 \
  python -m forge.forge_verify --in cosyn_qa.jsonl --image-root /path/imgs \
    --out forge_cosyn.jsonl --config forge_cosyn --mode debate --pass-source --workers 32
```

**A/B pilot.** Run once with `--mode single` and once with `--mode debate` over the same ~150–200
pairs, then score kept-answer **precision** against a hand-checked reference (and note **yield**).
Include the structured source in the input rows (`data` field) so the numeric gate and the judges
are grounded; without it, numeric answers simply fall through to the panel (more calls, same logic).
