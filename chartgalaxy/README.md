# ChartGalaxy QA-generation pipeline (real, English, image-verified)

Generates InfoVQA enrichment data from **real** ChartGalaxy infographics: English-only, with
questions/answers **generated and verified by Qwen3.6-27B-FP8 in image mode**. Question types
follow the grounded taxonomy in [`qa_spec.md`](qa_spec.md)
(DVQA / PlotQA / FigureQA / ChartQA).

## Stages

| Stage | Script | GPU? | Output |
|---|---|---|---|
| 1. fetch + English filter | `download_real_en.py` | no | `chartgalaxy_real_en/images/*.jpg` + `real_en_meta.jsonl` |
| — serve the VLM | `serve_qwen_vllm.slurm` | 2 GPU | OpenAI endpoint on `:8000` |
| 2. generate + verify QA | `gen_qa.py` | via endpoint | `chartgalaxy_real_en/real_en_qa.jsonl` |

Data lives under `/home/reza/data/chartgalaxy_real_en/`; code lives here.

## Run

```bash
# 1) fetch English real infographics (langdetect en>0.90, then download from source URLs)
python data_enrichment/chartgalaxy/download_real_en.py --limit 20000

# 2) serve the verifier/generator VLM (prints the node + :8000)
sbatch data_enrichment/chartgalaxy/serve_qwen_vllm.slurm

# 3) generate 3 QA/image, image-verified, against that endpoint
QA_LLM_BASE_URL=http://<node>:8000/v1 \
  python data_enrichment/chartgalaxy/gen_qa.py --num-questions 3 --workers 8
```

## How it works

- **English filter** — `langdetect` on each chart's title/label text, keep `en` prob > 0.90.
  Real ChartGalaxy is ~60-72% English; the rest (mostly DE/FR/ES) and number-only tables drop.
- **Generation** — the VLM sees the **image + table + chart_type** and returns up to 3 diverse
  pairs as JSON, spanning structural / data-retrieval / reasoning / visual tiers.
- **Verification (image-based)** — every pair goes back to the VLM **with the image**: "is this
  answer correct for this chart?" Confirmed pairs are kept; if rejected with a corrected answer,
  the corrected one is kept; otherwise dropped. Image-based (not table-based) because real
  infographics render values/labels the table may omit.
- **Resumable** — both stages skip already-processed ids/images, so re-running continues.

Output rows are ready for the mixed trainer: `{config: "chartgalaxy_real_en", image (abs path),
question, answer}`.

## Status

- Stage 1 validated (fetches English images + metadata).
- Stage 2 mechanics validated (image encoding, JSON parse, verification loop); needs the vLLM
  endpoint (stage "serve") running to generate at scale.
