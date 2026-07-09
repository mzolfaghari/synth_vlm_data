"""Prompts for FORGE's adversarial adjudication (Stage 4).

Adapted from the asymmetric-debate design of Mazza & Levi, "Synthetic Training of Custom Policy
Guardrails via Asymmetric Debate" (arXiv:2604.25203), Appendix A, retargeted from text policy
classification to VQA answer-correctness. The target "label" here is *"the proposed answer is
correct"*: a rigid Advocate defends it, while two judges with opposite biases (recall vs.
precision) independently decide by looking at the image. A pair is accepted only on consensus.
"""

_JSON = ('Return ONLY JSON: {"label":"correct|incorrect","confidence":0-100,'
         '"reason":"one sentence","corrected_answer":"<short answer, or empty if correct>"}')


def _ctx(context):
    return (f"\nThe image was rendered from this structured source (ground truth):\n{context}\n"
            if context else "\n")


# ---------- Advocate (rigid proponent) ----------
def advocate_system():
    return ("You are the ADVOCATE on a panel reviewing a visual-question-answering pair. "
            "You will see an IMAGE, a QUESTION, and a PROPOSED ANSWER. Argue, using only what is "
            "visible in the image, that the PROPOSED ANSWER is correct. You are a rigid proponent: "
            "make the strongest honest case for the answer and never concede. Be concise (2-3 "
            "sentences).")


def advocate_user(question, answer, context=None):
    return (f"QUESTION: {question}\nPROPOSED ANSWER: {answer}{_ctx(context)}"
            "Argue why the proposed answer is correct for this image.")


# ---------- Judges (independent, opposite biases) ----------
_PERSONA = {
    "recall": ("Persona: prioritise RECALL. Accept the answer as correct if it is reasonably "
               "supported by the image, tolerating minor wording or formatting differences."),
    "precision": ("Persona: be STRICT, favouring PRECISION with no interpretive leeway. Accept "
                  "only if the answer is exactly and unambiguously correct given the image; reject "
                  "on any inaccuracy, ambiguity, or over-claim."),
}


def judge_system(persona):
    return ("You are a JUDGE on a panel deciding whether a PROPOSED ANSWER to a visual QUESTION is "
            "correct, by looking at the IMAGE. " + _PERSONA[persona] + " If it is incorrect, give "
            "the correct short answer in corrected_answer. " + _JSON)


def judge_user_round1(question, answer, context, advocate_arg):
    return (f"QUESTION: {question}\nPROPOSED ANSWER: {answer}{_ctx(context)}"
            f"The Advocate argues: {advocate_arg}\n"
            "Independently decide whether the proposed answer is correct.")


def judge_user_subsequent(question, answer, context, advocate_arg, own_prev, others_prev):
    others = "\n".join(f"- another judge said: {json_dumps(o)}" for o in others_prev)
    return (f"QUESTION: {question}\nPROPOSED ANSWER: {answer}{_ctx(context)}"
            f"The Advocate argues: {advocate_arg}\n"
            f"Your previous response: {json_dumps(own_prev)}\n{others}\n"
            "At least one judge disagreed. Reconsider carefully. This does NOT mean you should "
            "automatically switch — weigh whether their reasoning is correct and relevant, then "
            "give your updated decision on whether the proposed answer is correct.")


# ---------- Single-judge baseline (for A/B vs. the panel) ----------
def single_judge_system():
    return ("You are an impartial judge. Looking at the IMAGE, determine whether the PROPOSED "
            "ANSWER to the QUESTION is correct. If incorrect, give the correct short answer. " + _JSON)


def single_judge_user(question, answer, context=None):
    return f"QUESTION: {question}\nPROPOSED ANSWER: {answer}{_ctx(context)}Is the proposed answer correct?"


def json_dumps(o):
    import json
    try:
        return json.dumps(o, ensure_ascii=False)
    except Exception:
        return str(o)
