# synth_vlm_data

Synthetic / augmented data-generation pipelines for vision-language model (VLM) SFT, built to
enrich chart & document understanding (InfoVQA / ChartQA-style tasks). Two pipelines:

| Dir | Pipeline | What it produces |
|---|---|---|
| [`chartgalaxy/`](chartgalaxy/) | **ChartGalaxy QA generation** — English, image-verified QA over real + synthetic chart infographics, generated *and* verified by a VLM (Qwen3.6-27B-FP8) in image mode. | `{config, image, question, answer, tier, rationale}` JSONL |
| [`cosyn/`](cosyn/) | **CoSyn-400K** download + manifest builder for text-rich synthetic images (doc / table / nutrition / chart / diagram / graphic / math). | `{config, image, question, answer}` JSONL |

## chartgalaxy — the QA generator

Generate + verify multi-tier QA (structural / data-retrieval / reasoning / visual) grounded in the
DVQA / PlotQA / FigureQA / ChartQA taxonomy. Quality controls: single-value answers, CoT rationale,
numeric groundedness (±5% vs the chart's data table), ROUGE-L dedup, and per-tier balance.

- [`chartgalaxy/README.md`](chartgalaxy/README.md) — quick start
- [`chartgalaxy/PIPELINE.md`](chartgalaxy/PIPELINE.md) — end-to-end stages (download → serve VLM → generate+verify)
- [`chartgalaxy/qa_spec.md`](chartgalaxy/qa_spec.md) — question taxonomy
- [`chartgalaxy/SOTA_pipelines.md`](chartgalaxy/SOTA_pipelines.md) — survey of comparable chart-QA pipelines
- [`chartgalaxy/gen_qa.py`](chartgalaxy/gen_qa.py) — the generate+verify client (OpenAI-compatible endpoint)

## cosyn — CoSyn-400K builder

- [`cosyn/download.py`](cosyn/download.py) — pull CoSyn-400K configs and materialize images
- [`cosyn/build_final.py`](cosyn/build_final.py) — normalize into a training manifest (3 QA/image cap)

## Notes

- **Code and docs only.** Generated JSONL, downloaded images, and sample charts are **not** tracked
  (see `.gitignore`) — regenerate them with the download/build scripts. The `.slurm` files carry the
  original cluster paths as reference; adapt roots/partitions to your environment.
- **Licensing:** ChartGalaxy's own license is unsettled (HF `cc-by-nc-4.0` vs GitHub `Apache-2.0`),
  and real infographics originate from third parties (e.g. Statista, Pew) — clear source terms
  before any commercial use of data these scripts fetch. This repo (the pipeline code) is Apache-2.0.
