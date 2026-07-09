#!/usr/bin/env python3
"""ChartGalaxy pipeline — STAGE 2: generate + verify QA with Qwen3.6-27B-FP8 (IMAGE mode).

For each English real infographic (from stage-1 real_en_meta.jsonl):
  1. GENERATE — send the IMAGE + underlying table + chart_type to the VLM and ask for up to
     --num-questions diverse QA pairs, covering the grounded taxonomy (structural / data
     retrieval / reasoning / visual — see qa_spec.md, grounded in DVQA, PlotQA,
     FigureQA, ChartQA).
  2. VERIFY — send the IMAGE + (question, proposed answer) back to the VLM and ask whether the
     answer is correct *by looking at the chart*. Keep pairs the model confirms; if it rejects
     but supplies a corrected answer, keep the corrected one. Otherwise drop.

Verification is IMAGE-based (not table-based) on purpose: real infographics often show values /
labels the table doesn't capture, so the rendered chart is the ground truth.

Output SFT rows {config, image, question, answer} -> real_en_qa.jsonl (resumable).

Serving: any OpenAI-compatible vLLM endpoint for Qwen3.6-27B-FP8 with vision (see
serve_qwen_vllm.slurm). Configure via --base-url / env QA_LLM_BASE_URL.

Usage:
  gen_qa.py [--limit N] [--num-questions 3] [--workers 8] \
            [--base-url http://localhost:8000/v1] [--model qwen3.6-27b-fp8]
"""
import os, io, json, sys, base64, argparse, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ROOT = "/home/reza/data/chartgalaxy_real_en"
META = f"{ROOT}/real_en_meta.jsonl"
OUT = f"{ROOT}/real_en_qa.jsonl"
MAX_SIDE = 1280  # downscale long side sent to the VLM (keeps text legible, bounds tokens)

GEN_SYS = (
    "You write training questions for a chart & infographic understanding model. "
    "You are given an infographic IMAGE and its underlying data table. Produce diverse, "
    "self-contained question-answer pairs that are answerable by LOOKING AT THE CHART. "
    "Cover different reasoning types from this taxonomy (grounded in DVQA / PlotQA / FigureQA / "
    "ChartQA):\n"
    "  - structural: count of series/bars, element identity, what the legend encodes\n"
    "  - data_retrieval: read one specific value for a category/series\n"
    "  - reasoning: extremum (highest/lowest), pairwise comparison (yes/no), aggregation "
    "(sum/avg/count-above), difference/ratio, trend over an ordered axis\n"
    "  - visual: reference a color / icon / position in the chart\n"
    "STRICT ANSWER FORMAT: every answer is a SINGLE number or a SINGLE word/short label — the "
    "bare value ONLY. Examples of valid answers: '50', '50%', '31 cm', '1.1M', 'Aptos', "
    "'Democrats', 'Educational institutions', 'yes', 'no'. NEVER a sentence, NEVER an explanation, "
    "NEVER restate the question, NEVER a list. Phrase each question so its answer is one "
    "value/word (\"Which X is highest?\" -> a label; \"What is X?\" -> a number). If the natural "
    "answer would be a list or sentence, rewrite the question to target a single value.\n"
    "Also: each question answerable from the IMAGE alone; do NOT mention 'the table'; no outside "
    "knowledge. Also give a 'rationale': ONE short sentence stating how the answer is read or "
    "computed from the chart (required for reasoning questions, brief for others).\n"
    'Return ONLY JSON: {"qa":[{"tier":"<one of above>","question":"...",'
    '"answer":"<single number or word>","rationale":"<one sentence>"}]}'
)
VERIFY_SYS = (
    "You verify a chart question-answer pair by LOOKING AT THE IMAGE. Report two things:\n"
    "  1. english: is the chart's PRINTED text (title, labels, captions) primarily English? "
    "Judge by the printed words, not by brand names or numbers.\n"
    "  2. correct: is the proposed answer correct for the question given the chart? If it is "
    "wrong but you can read the correct answer from the chart, provide it in corrected_answer.\n"
    'Return ONLY JSON: {"english": true/false, "correct": true/false, '
    '"corrected_answer": "<exact answer or empty>"}'
)

def b64_image(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size; m = max(w, h)
    if m > MAX_SIDE:
        s = MAX_SIDE / m; im = im.resize((int(w*s), int(h*s)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()

def chat(base_url, model, system, user_text, img_b64, temperature, max_tokens=2048,
         timeout=300, retries=3):
    msg = [{"role": "system", "content": system},
           {"role": "user", "content": [
               {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
               {"type": "text", "text": user_text}]}]
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(f"{base_url.rstrip('/')}/chat/completions",
                              json={"model": model, "messages": msg, "temperature": temperature,
                                    "max_tokens": max_tokens,
                                    # Qwen3 is a reasoning model; chart QA is perception, not multi-
                                    # step reasoning — disable "thinking" so calls are fast/reliable.
                                    "chat_template_kwargs": {"enable_thinking": False}},
                              timeout=timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise last

def parse_json(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None

def norm_answer(a):
    return str(a).strip().rstrip(". ").strip()

def ok_answer(a):
    # single value / short label / yes-no only — reject sentences and lists.
    # Applied to BOTH the generated answer and the verifier's corrected_answer.
    return bool(a) and len(a.split()) <= 8 and a.count(",") < 2

# ── #2 numeric groundedness (ChartQA-style ±5%): a lookup answer's number must appear in table ──
def parse_num(s):
    s = str(s).strip().lower().replace(",", "")
    m = re.search(r"-?\d*\.?\d+", s.replace("$", "").replace("%", ""))
    if not m: return None
    v = float(m.group(0))
    suf = re.search(r"\d\s*([kmb])(?![a-z])", s)   # magnitude suffix (1.1M) but not units (cm/km)
    if suf: v *= {"k": 1e3, "m": 1e6, "b": 1e9}[suf.group(1)]
    return v

def table_numbers(data):
    nums = []
    def w(x):
        if isinstance(x, bool): return
        if isinstance(x, (int, float)): nums.append(float(x))
        elif isinstance(x, str):
            for tok in re.findall(r"-?\d[\d,]*\.?\d*\s*[kmbKMB]?", x):  # keep magnitude suffix
                v = parse_num(tok)
                if v is not None: nums.append(v)
        elif isinstance(x, list):
            for e in x: w(e)
        elif isinstance(x, dict):
            for e in x.values(): w(e)
    try: w(json.loads(data) if isinstance(data, str) else data)
    except Exception: pass
    return nums

def grounded_numeric(answer, tnums, tol=0.05):
    """True unless the answer is numeric, the table has numbers, and none match within ±5%."""
    av = parse_num(answer)
    if av is None or not tnums: return True
    for t in tnums:
        if (abs(av) < 1e-9 and abs(t) < 1e-9) or (t != 0 and abs(av - t) / abs(t) <= tol):
            return True
    return False

# ── #3 dedup: ROUGE-L (token-LCS F1); reject near-duplicate questions per image (Self-Instruct) ──
def _lcs(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def rouge_l(x, y):
    xt, yt = x.lower().split(), y.lower().split()
    if not xt or not yt: return 0.0
    l = _lcs(xt, yt)
    if l == 0: return 0.0
    p, r = l / len(yt), l / len(xt)
    return 2 * p * r / (p + r)

def process(rec, base_url, model, n):
    try:
        img = b64_image(rec["image"])
    except Exception:
        return []
    tnums = table_numbers(rec.get("data"))              # #2: numbers present in the chart's table
    table = json.dumps(rec.get("data"), ensure_ascii=False)[:4000]
    gen_user = (f"Chart type: {rec.get('chart_type')}. Underlying data table:\n{table}\n\n"
                f"Generate {n} diverse question-answer pairs as specified.")
    try:
        gen = parse_json(chat(base_url, model, GEN_SYS, gen_user, img, 0.7))
    except Exception:
        return []
    if not gen or "qa" not in gen: return []
    LOOKUP = ("data_retrieval", "structural")           # tiers whose answer should be a raw cell value
    kept = []; tier_count = {}
    for qa in gen["qa"][:n]:
        q = str(qa.get("question", "")).strip()
        a = norm_answer(str(qa.get("answer", "")))
        tier = str(qa.get("tier", "")).strip().lower()
        rationale = str(qa.get("rationale", "")).strip()
        if not q or not ok_answer(a): continue
        if tier in LOOKUP and not grounded_numeric(a, tnums):   # #2 drop fabricated lookup numbers
            continue
        if any(rouge_l(q, k["question"]) >= 0.7 for k in kept): continue   # #3 dedup near-dup questions
        if tier_count.get(tier, 0) >= 2: continue                          # #3 tier balance (<=2/tier)
        vuser = f"Question: {q}\nProposed answer: {a}\nIs the proposed answer correct?"
        try:
            v = parse_json(chat(base_url, model, VERIFY_SYS, vuser, img, 0.0, max_tokens=1024))
        except Exception:
            continue
        if not v: continue
        if v.get("english") is False:      # non-English chart (image-judged) -> drop whole image
            return []
        if v.get("correct") is True:
            final_a = a
        else:
            ca = norm_answer(str(v.get("corrected_answer", "")))
            if not ok_answer(ca): continue                       # guard corrected answer
            if tier in LOOKUP and not grounded_numeric(ca, tnums): continue
            final_a = ca
        kept.append({"config": rec["config"], "image": rec["image"], "question": q,
                     "answer": final_a, "tier": tier, "rationale": rationale})
        tier_count[tier] = tier_count.get(tier, 0) + 1
    return kept

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("QA_LLM_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("QA_LLM_MODEL", "Qwen/Qwen3.6-27B-FP8"))
    ap.add_argument("--num-questions", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all in meta")
    ap.add_argument("--meta", default=META, help="input metadata jsonl (stage-1 output)")
    ap.add_argument("--out", default=OUT, help="output QA jsonl (resumable). Use a new path for a v2 re-run.")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.meta)]
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try: done.add(json.loads(l)["image"])
            except Exception: pass
    recs = [r for r in recs if r["image"] not in done]
    if args.limit: recs = recs[:args.limit]
    print(f"to process: {len(recs)} images  (endpoint {args.base_url}, model {args.model}, out {args.out})", flush=True)

    out = open(args.out, "a"); n_pairs = 0; n_img = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, r, args.base_url, args.model, args.num_questions): r for r in recs}
        for fut in as_completed(futs):
            rows = fut.result() or []
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n"); n_pairs += 1
            out.flush(); n_img += 1
            if n_img % 100 == 0:
                sys.stderr.write(f"imgs={n_img}/{len(recs)} pairs={n_pairs} "
                                 f"({n_pairs/max(n_img,1):.2f}/img, {n_img/max(time.time()-t0,1):.1f} img/s)\n")
                sys.stderr.flush()
    out.close()
    print(f"DONE images={n_img} verified_pairs={n_pairs} -> {args.out}")

if __name__ == "__main__":
    main()
