# SOTA synthetic chart-QA pipelines — findings & what to adopt

Survey of published chart/infographic instruction-data pipelines, to align our ChartGalaxy
generator (`gen_qa.py`) with best practice. Our pipeline = per real image: VLM(image+table) →
3 QA across structural/data-retrieval/reasoning/visual → 2nd VLM call verifies each answer
against the image + English gate → answers constrained to a single value/word.

## Sources

| Pipeline | arXiv | Generation | QA generator | Quality control | Code | Answer format |
|---|---|---|---|---|---|---|
| **ChartGalaxy** | 2505.18668 | retrieval-aug (SBERT→Gemini), from table | Gemini | described only | **no QA/code released** | short |
| **ChartLlama** | 2311.16483 | code-guided: GPT-4 table→matplotlib→QA | GPT-4 (text, from table) | **light** — drop only if plot code fails | [code](https://github.com/tingxueronghua/ChartLlama-code)+data | free-form + reasoning |
| **ChartX/ChartVLM** | 2402.12185 | code-guided GPT-4 table→render→cognition tasks | GPT-4 | **4-step human validation** (null → cross-modal align → per-QA reasoning-step → GPT-scored); **QA scored Exact-Match ±5% numeric** | [code](https://github.com/UniModal4Reasoning/ChartVLM)+[data](https://huggingface.co/datasets/U4R/ChartX) | task-dep: value+5% / CSV / code |
| **ChartInstruct** | 2403.09028 | LLM-from-table (Gemini extracts table for web charts) | GPT-3.5 (simple) / GPT-4 (CoT, code) | expert audit of 100 (87% valid task, 61% fully-correct output); **keeps imperfect outputs**; ViT chart/non-chart filter | [code](https://github.com/vis-nlp/ChartInstruct) (model-centric; prompts in appendix) | mostly free-form/CoT; **CoT ends "The Answer is X"** |
| **CoSyn-400K** | allenai | code-guided synth | GPT/Claude from data | grounded in code/data | released | **ships `explanation` (rationale) + `answer`** |
| **ChartQA** | 2203.10244 | human split + **machine split from tables** (T5) | humans + T5 | human split hand-written | [code](https://github.com/vis-nlp/ChartQA) | short value/label |
| **PlotQA / DVQA** | 1909.00997 | templates over table | templates+paraphrase | deterministic | released | single value |

## The consensus (what makes SOTA data trustworthy)

1. **Answers are grounded in the known table/code, not inferred from pixels.** Every grounded
   pipeline (ChartLlama, ChartX, CoSyn, ChartQA-machine, PlotQA) generates answers from the data
   they control. **None uses multi-model / k-vote verification** — table-grounding replaces it.
2. **Numeric answers checked with tolerance** (ChartX: Exact-Match ±5%), not exact string.
3. **Reasoning questions carry a rationale/CoT** (CoSyn `explanation`; ChartInstruct CoT), with a
   **minimal terminal answer** for exact-match ("The Answer is X").
4. **Task/question diversity** via distinct task templates (ChartX 7 tasks, ChartInstruct 6).
5. QC rigor varies: ChartX (heavy human) ↔ ChartLlama/ChartInstruct (light, tolerate noise).

## What our pipeline is missing — ranked

1. **Numeric groundedness check (±5%)** — *highest impact.* SOTA grounds answers in the table.
   We can't fully recompute (our questions are LLM-freeform, so we don't know which cell each maps
   to), but the practical form: for **data-retrieval / structural** tier answers that are numeric,
   require the number to appear among the table's numeric values within ±5% — else drop
   (catches fabricated values). Skip for **reasoning** tier (difference/ratio → computed, not a raw
   cell). Uses the `tier` field we already emit. Cheap, no extra model.
2. **Question-tier diversity + dedup** — enforce the 3 questions span **distinct tiers** and drop
   near-duplicate questions per image (normalize + fuzzy). We prompt for diversity but don't enforce.
3. **CoT rationale field for reasoning questions** — add `rationale` alongside the single-value
   `answer` (CoSyn/ChartInstruct convention). Keeps exact-match eval; gives stronger reasoning
   training signal, and enables a reasoning-step check like ChartX.
4. **Difficulty tag** per pair (tier already ~= this) for balanced sampling later. Low effort.
5. **k-vote / multi-model verify** — *skip.* SOTA doesn't use it; table-grounding is better ROI and
   matches the "no extra verifier" decision.

## Round-2 corrections & additions (full agent sweep)

**Correction — ChartGalaxy's QA method is essentially OURS.** Per Appendix I.1: Gemini-2.0-Flash
is given **both the table AND the chart image** to produce concise QA (a "number, text label, or
Yes/No found directly in the data"). Retrieval-augmentation (SBERT) is used only for chart
*title* generation, **not** QA. Some pairs are template-based, **adapted from ChartAssistant's
templates**. Only the 4,975-pair eval set is human-verified; the 443K train set relies on prompt
constraints ("follow the table strictly; don't contradict it"). → **We are already aligned with
the most relevant paper** (image+table → VLM → concise answers, ±5% relaxed-accuracy eval).

**ChartAssistant / ChartSFT (2401.02384)** — [code](https://github.com/OpenGVLab/ChartAst).
Numeric & referring QA come from **101 + 114 hand-written templates** (answers ground-truth by
construction). Open QA from ChatGPT-on-table. **Numeric answers annotated as JSON chain-of-thought
(retrieve-step + compute-step); CoT lifted numeric-QA accuracy 51.9 → 72.1%** — strong, measured
evidence for rationales on numeric/reasoning questions.

**MMC (2311.10774)** — [code](https://github.com/FuxiaoLiu/MMC). GPT-4-from-text; **answers capped
<20 words** for anti-hallucination (we already do this); human validation of 500 (91% appropriate,
85% acceptable). Free-form + MQA.

**ChartQA groundedness filter** — the simple, proven QC: *"filter out the question if the answer
cannot be found in the chart data table."* This is the pragmatic form of #1.

**Revised priority given the sweep:**
- Our core (image+table→VLM→concise, image-verify) = ChartGalaxy's method. Good baseline.
- **CoT rationale for numeric/reasoning questions moves UP** — ChartAssistant's +20% accuracy is
  the strongest measured result in the survey. Store `answer` (single value) + `rationale`.
- **Numeric groundedness gate** (ChartQA-style: answer value must appear in the table, ±5%) — cheap.
- Diversity/dedup — still worthwhile.

**ChartBench (2312.15915)** — [code](https://github.com/IDEA-FinAI/ChartBench). A *benchmark*
(least applicable). Two ideas: (a) **paired positive/negative yes-no assertions** — model must
answer both a correct and an incorrect variant right (Acc+), defeating lucky guesses — an
eval-metric idea, not generation; (b) **prefer *unannotated* charts** (values must be read from
bars/axes, not printed labels) as harder signal. Reinforces tolerance-based numeric scoring + CoT.

## Concrete verification techniques (final sweep) + authoritative answer metrics

**Groundedness/verification techniques, by strength (with citations):**
1. **Answer from table/code, not pixels** — CoSyn (2502.14846), ChartLlama, ChartAssistant. Structural guarantee.
2. **Template/backend-computed numerics** — ChartAssistant (101 templates, backend executes).
3. **PoT execution round-trip** — **TinyChart (2404.16635)**: generate a Python program for the
   answer, execute it, and **keep the pair only if the result matches the gold answer**; drop if it
   errors or mismatches. The strongest *re-derivation* filter; concrete and copyable.
   [code](https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/TinyChart)
4. **LLM-as-judge confidence filter** — **ECD (2508.06492)**: GPT-4o rates each QA 1–5, keep only 5
   (dropped ~7.8%). [code](https://github.com/yuweiyang-anu/ECD)
5. **Multi-sample self-consistency** — Chart-CoCa (2508.11975). (inference-time; skip for us)

**Authoritative answer-format / scoring conventions (what the benchmarks actually use):**
- **ChartQA / PlotQA / InfoVQA(numeric):** *relaxed accuracy* = **5% relative tolerance for numeric,
  exact match for text**. → our single-value answers are correct; use ±5% for any numeric check.
- **DocVQA / InfographicVQA(text):** ANLS (edit-distance, τ=0.5). InfographicVQA uniquely allows
  **multi-span/list answers** (comma-separated, all permutations valid) — the one place a list
  answer is legitimate.
- Surveys to cite: **2403.12027** (From Pixels to Insights, 7-task taxonomy + metrics) and ECD.

**Unanswerable questions:** essentially absent from synthetic *training* pipelines — classic sets
(InfographicVQA) *removed* the ~1.9% flagged unanswerable. Only **ChartQAPro (2504.05506)** (eval)
deliberately includes them (gold = "unanswerable"). Our guard already drops "chart does not
provide…" — consistent with the norm. Teaching abstention is optional/advanced (recipe: UNK-VQA
2310.10942, SQuAD-2.0 1806.03822). Low priority.

**Dedup thresholds (concrete) & difficulty:** per-question dedup isn't done by chart papers, but the
transferable recipes are Self-Instruct **ROUGE-L < 0.7** (cheapest, per-image), Deita Repr-Filter
**cosine distance < 0.90** (embedding), and SemDeDup (k-means + cosine). Docmatix precedent: 5 prompts
+ discard ~15% flagged incorrect. Difficulty axis everyone uses = **descriptive vs reasoning**
(CharXiv 2406.18521, ~4 descriptive:1 reasoning/chart) — our `tier` field already encodes this;
balancing the per-image mix toward that ratio is the low-effort version.

## Feasibility caveat

Full deterministic table-verification (like ChartLlama/ChartX) is easier for them because they
**generate QA from templates with known cell references**. We generate freeform QA from an LLM, so
we can only do the *groundedness* form (#1: is the answer value present in the table ±5%) for
retrieval-tier answers, plus the existing image-based Qwen verify for everything else. That's the
right adaptation given our freeform generation.
