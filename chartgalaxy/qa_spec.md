# ChartGalaxy QA-generation spec (grounded in published taxonomies)

Question-generation design for the ChartGalaxy InfoVQA enrichment. Every question type here is
grounded in a **published, cited** chart-QA dataset — not ad-hoc templates.

## Provenance check (why this doc exists)

**ChartGalaxy did NOT release QA templates or generation code.** Verified against its GitHub tree
(122 files): only `code_generation_benchmark/` (chart→code), `code_understanding/`,
`example_based_generation/` (making new infographics), `examples/`. No VQA/instruction/QA
generation code. The paper (arXiv:2505.18668) describes its 443K-pair set only *verbally*:
text-based reasoning (data identification, comparison, extraction-with-condition, fact-checking),
visual-element reasoning, and visual understanding — generated with retrieval-augmented Gemini
prompting. No artifacts to reuse.

So we ground our taxonomy on the datasets ChartGalaxy itself cites as "template-based questions
from prior work."

## Grounded taxonomy

Both **DVQA** (arXiv:1801.08163) and **PlotQA** (arXiv:1909.00997) independently define the same
three tiers; **FigureQA** (arXiv:1710.07300) and **ChartQA** (arXiv:2203.10244) add the relational
and visual/arithmetic types. Each row cites its source.

| Tier | Question type | Template (example) | Source |
|---|---|---|---|
| **1. Structural** | count elements | "How many {series/bars} are shown?" | DVQA, PlotQA |
| | element identity | "What is the label of the {N}th bar from the left?" | DVQA |
| | legend/encoding | "What does the legend indicate?" | PlotQA |
| **2. Data retrieval** | single-value lookup | "What is the {metric} for {category}?" | DVQA, PlotQA, ChartQA |
| | series+axis lookup | "What is {series}'s {metric} in {year}?" | PlotQA |
| **3. Reasoning** | extremum (max/min) | "Which {category} has the highest {metric}?" | DVQA, PlotQA, FigureQA |
| | pairwise comparison (yes/no) | "Is {A}'s {metric} greater than {B}'s?" | FigureQA |
| | aggregation | "In how many {categories} is {metric} above the average?" | PlotQA (conditional template) |
| | difference / ratio | "What is the difference between {A} and {B}?" | ChartQA (arithmetic) |
| | trend over ordered axis | "Did {metric} increase or decrease from {t0} to {t1}?" | ChartQA |
| **4. Visual / compositional** | color/icon/position reference | "What does the {color} segment represent?" | ChartQA (human split), ChartGalaxy (visual-element) |

Note: my original sample templates (`argmax / lookup / difference / trend`) map onto tiers 2–3 —
they were a subset of this, but invented rather than cited. This doc supersedes them.

## Answer format (STRICT — matches DVQA / PlotQA / ChartQA / InfoVQA)

Every answer must be a **single number or a single word/short label** — the bare value only,
never a sentence or explanation. This is what makes the pairs exact-match scorable, exactly like
the benchmarks:

- **number**: `50`, `50%`, `1.1M`, `31 cm`, `2257` (include the unit/% only if the chart shows it)
- **single word / short entity label**: `Aptos`, `Democrats`, `Toyota` — the exact label as drawn
  (a multi-word proper label like `Educational institutions` or `United States` is fine; a
  *sentence* is not)
- **yes / no** for comparison questions

Rules enforced at generation:
- The question must be phrased so its answer is one value/word (e.g. "Which X is highest?" → a
  label; "What is X?" → a number).
- The answer field contains ONLY that value — no restating the question, no "because…", no
  "…selected by 50% of respondents".
- Reject list answers and phrase answers; if the natural answer is a list/sentence, rewrite the
  question to target a single value instead.

Bad → `"Increasing brand awareness is the top priority, selected by 50% of respondents."`
Good → Q: "Which priority ranks highest?" A: `Increasing brand awareness`  (or Q: "What % chose the
top priority?" A: `50%`)

## Answer generation — table-verified

Answers are **derived from the structured table**, never free-generated:

1. **LLM drafts** the question + candidate answer, given: the `data` table, `chart_type`, column
   descriptions, and title. (LLM handles column-role reasoning — e.g. Year/Month are temporal
   axes, not metrics — and multi-series structure that pure templates get wrong.)
2. **Verification gate**: for any numeric/lookup/extremum/aggregation answer, **recompute from the
   table** and drop the pair if the LLM's answer disagrees. Structural/visual answers are checked
   against `chart_type` metadata.
3. Only verified pairs are emitted.

This mirrors ChartGalaxy's own LLM-based method while keeping labels provably correct — the
advantage of a table-backed source over scraped QA.

## Practical defaults

- Question mix per chart: ~1 structural, ~1 retrieval, ~1–2 reasoning, 0–1 visual (skip visual if
  the chart has no color/icon legend). Target ~3 usable pairs/chart after verification.
- **English only** — `langdetect` on the table text, `en` prob > 0.90 (real ChartGalaxy is
  ~60–72% English; synthetic is all English).
- Exclude number-only tables from structural/visual questions (no labels to reference).

## Sources

- DVQA — Kafle et al., arXiv:1801.08163
- PlotQA — Methani et al., arXiv:1909.00997
- FigureQA — Kahou et al., arXiv:1710.07300
- ChartQA — Masry et al., arXiv:2203.10244
- ChartGalaxy — Li et al., arXiv:2505.18668 (taxonomy described; no templates/code released)
