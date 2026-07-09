"""Cheap sanity filters — the first, no-LLM stage (Stage 3 of FORGE).

These gates only DROP obviously-bad candidates (not a single-value answer, or a near-duplicate of
one already kept for the image). They do NOT judge correctness — every surviving pair goes to the
adversarial panel (Stage 4), which is the sole arbiter of correctness. Helpers are ported from
chartgalaxy/gen_qa.py (answer-format guard, ROUGE-L dedup, and numeric ±5% groundedness which is
still used by the panel's refinement step to sanity-check corrected answers).

`triage()` returns one of:
  ("drop",       reason)   -> junk (bad format / near-duplicate)      -> discard
  ("adjudicate", "passes") -> everything else                         -> send to the panel
"""
import re, json


def norm_answer(a):
    return str(a).strip().rstrip(". ").strip()


def ok_answer(a):
    # single value / short label / yes-no only — reject sentences and lists.
    return bool(a) and len(a.split()) <= 8 and a.count(",") < 2


def parse_num(s):
    s = str(s).strip().lower().replace(",", "")
    m = re.search(r"-?\d*\.?\d+", s.replace("$", "").replace("%", ""))
    if not m:
        return None
    v = float(m.group(0))
    suf = re.search(r"\d\s*([kmb])(?![a-z])", s)   # magnitude suffix (1.1M) but not units (cm/km)
    if suf:
        v *= {"k": 1e3, "m": 1e6, "b": 1e9}[suf.group(1)]
    return v


def source_numbers(data):
    """Every number appearing in the image's structured source (data table / render code)."""
    nums = []
    def w(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            nums.append(float(x))
        elif isinstance(x, str):
            for tok in re.findall(r"-?\d[\d,]*\.?\d*\s*[kmbKMB]?", x):
                v = parse_num(tok)
                if v is not None:
                    nums.append(v)
        elif isinstance(x, list):
            for e in x:
                w(e)
        elif isinstance(x, dict):
            for e in x.values():
                w(e)
    try:
        w(json.loads(data) if isinstance(data, str) else data)
    except Exception:
        pass
    return nums


def grounded_numeric(answer, snums, tol=0.05):
    """True unless the answer is numeric, the source has numbers, and none match within ±5%."""
    av = parse_num(answer)
    if av is None or not snums:
        return True
    for t in snums:
        if (abs(av) < 1e-9 and abs(t) < 1e-9) or (t != 0 and abs(av - t) / abs(t) <= tol):
            return True
    return False


def _lcs(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def rouge_l(x, y):
    xt, yt = x.lower().split(), y.lower().split()
    if not xt or not yt:
        return 0.0
    l = _lcs(xt, yt)
    if l == 0:
        return 0.0
    p, r = l / len(yt), l / len(xt)
    return 2 * p * r / (p + r)


def triage(question, answer, kept_questions, dedup_thresh=0.7):
    """Cheap sanity filters, no LLM. Drop junk; send everything else to the panel — the panel
    (not the gates) decides correctness. Returns ('drop', reason) or ('adjudicate', 'passes')."""
    a = norm_answer(answer)
    if not ok_answer(a):
        return ("drop", "bad_format")           # not a single short value (list/sentence)
    if any(rouge_l(question, kq) >= dedup_thresh for kq in kept_questions):
        return ("drop", "near_duplicate")
    return ("adjudicate", "passes")
