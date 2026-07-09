"""Asymmetric adversarial adjudication (Stage 4) + dissent-driven refinement.

`adjudicate()` is the entry point: a rigid Advocate defends the proposed answer while two judges
(recall + precision personas) independently decide over up to T rounds. Accept on consensus that
the answer is correct; on dissent, take the judges' corrected answer, refine, and re-adjudicate up
to R_max times, else discard. `single_verify()` is the one-judge baseline for A/B comparison.

All LLM calls go through llm.chat (imported at module level so tests can monkeypatch forge.debate.chat).
"""
from .llm import chat, parse_json
from .gates import norm_answer, ok_answer, parse_num, grounded_numeric
from . import prompts


def _is_correct(resp):
    if not resp:
        return False
    if resp.get("correct") is True:
        return True
    return str(resp.get("label", "")).strip().lower().startswith("correct")


def run_debate(base_url, model, img_b64, question, answer, context=None, rounds=2,
               personas=("recall", "precision"), temperature=0.1, max_tokens=500):
    """One debate over a fixed (question, answer). Returns dict with keys:
       kept(bool), rounds_used, per_round(list), objections(list), corrections(list)."""
    adv = chat(base_url, model, prompts.advocate_system(),
               prompts.advocate_user(question, answer, context), img_b64, 0.2, max_tokens)
    prev = {p: None for p in personas}
    per_round = []
    for t in range(rounds):
        resp = {}
        for p in personas:
            sys = prompts.judge_system(p)
            if t == 0:
                user = prompts.judge_user_round1(question, answer, context, adv)
            else:
                others = [prev[pp] for pp in personas if pp != p]
                user = prompts.judge_user_subsequent(question, answer, context, adv, prev[p], others)
            resp[p] = parse_json(chat(base_url, model, sys, user, img_b64, temperature, max_tokens)) or {}
        prev = resp
        per_round.append(resp)
        if all(_is_correct(resp[p]) for p in personas):
            return {"kept": True, "rounds_used": t + 1, "per_round": per_round,
                    "objections": [], "corrections": []}
    # dissent: collect objections + proposed corrected answers from the judges who rejected
    objections, corrections = [], []
    for p in personas:
        if not _is_correct(prev[p]):
            if prev[p].get("reason"):
                objections.append(f"{p}: {prev[p]['reason']}")
            ca = norm_answer(prev[p].get("corrected_answer", ""))
            if ca:
                corrections.append(ca)
    return {"kept": False, "rounds_used": rounds, "per_round": per_round,
            "objections": objections, "corrections": corrections}


def adjudicate(base_url, model, img_b64, question, answer, context=None, snums=None,
               rounds=2, r_max=1, personas=("recall", "precision")):
    """Gates already decided this pair needs the panel. Debate, then refine on dissent up to r_max.
       Returns dict: kept, answer(final), refined(bool), rounds_used, objections."""
    snums = snums or []
    cur = norm_answer(answer)
    for attempt in range(r_max + 1):
        res = run_debate(base_url, model, img_b64, question, cur, context, rounds, personas)
        if res["kept"]:
            return {"kept": True, "answer": cur, "refined": attempt > 0,
                    "rounds_used": res["rounds_used"], "objections": []}
        # pick a corrected answer that also passes the cheap guards (format + numeric grounding)
        cand = [c for c in res["corrections"]
                if ok_answer(c) and (parse_num(c) is None or grounded_numeric(c, snums))]
        if not cand or attempt == r_max:
            return {"kept": False, "answer": cur, "refined": attempt > 0,
                    "rounds_used": res["rounds_used"], "objections": res["objections"]}
        cur = cand[0]   # refine and re-adjudicate


def single_verify(base_url, model, img_b64, question, answer, context=None, max_tokens=500):
    """One-judge baseline (Appendix-A.4 style) for comparing against the panel."""
    r = parse_json(chat(base_url, model, prompts.single_judge_system(),
                        prompts.single_judge_user(question, answer, context), img_b64, 0.1, max_tokens)) or {}
    if _is_correct(r):
        return {"kept": True, "answer": norm_answer(answer), "refined": False, "objections": []}
    ca = norm_answer(r.get("corrected_answer", ""))
    if ok_answer(ca):   # accept a single-model correction (mirrors chartgalaxy's corrected_answer)
        return {"kept": True, "answer": ca, "refined": True, "objections": [r.get("reason", "")]}
    return {"kept": False, "answer": norm_answer(answer), "refined": False,
            "objections": [r.get("reason", "")]}
