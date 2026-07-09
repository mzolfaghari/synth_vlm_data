#!/usr/bin/env python3
"""Fold ChartGalaxy QA into the latest manifest, two versions:
  A (all)   = base + ChartGalaxy synthetic + ChartGalaxy real(v2)
  B (synth) = base + ChartGalaxy synthetic
ChartGalaxy rows are normalized to {config,image,question,answer} (tier/rationale dropped),
matching the base manifest. Single streaming pass (base read once).
"""
import json

BASE = "/home/reza/data/KUB2M_3Bsynth_MathEnrich_OCRDoc_ChartCoSyn.jsonl"
SYNTH = "/home/reza/data/chartgalaxy_synth/synth_qa.jsonl"
REAL = "/home/reza/data/chartgalaxy_real_en/real_en_qa_v2.jsonl"
OUT_ALL = "/home/reza/data/KUB2M_3Bsynth_MathEnrich_OCRDoc_ChartCoSyn_ChartGalaxyAll.jsonl"
OUT_SYNTH = "/home/reza/data/KUB2M_3Bsynth_MathEnrich_OCRDoc_ChartCoSyn_ChartGalaxySynth.jsonl"

KEYS = ("config", "image", "question", "answer")
def norm(line):
    o = json.loads(line)
    return json.dumps({k: o.get(k) for k in KEYS}, ensure_ascii=False) + "\n"

nb = ns = nr = 0
with open(OUT_ALL, "w") as gA, open(OUT_SYNTH, "w") as gB:
    # base -> both (pass-through; already 4-key)
    for line in open(BASE):
        if not line.strip(): continue
        out = line if line.endswith("\n") else line + "\n"
        gA.write(out); gB.write(out); nb += 1
    # synthetic ChartGalaxy -> both (normalized)
    for line in open(SYNTH):
        if not line.strip(): continue
        out = norm(line)
        gA.write(out); gB.write(out); ns += 1
    # real ChartGalaxy (v2) -> A only (normalized)
    for line in open(REAL):
        if not line.strip(): continue
        gA.write(norm(line)); nr += 1

res = {"base": nb, "chartgalaxy_synth": ns, "chartgalaxy_real_v2": nr,
       "A_all_total": nb + ns + nr, "B_synth_total": nb + ns,
       "OUT_ALL": OUT_ALL, "OUT_SYNTH": OUT_SYNTH}
print(json.dumps(res, indent=1))
json.dump(res, open("/home/reza/data/build_with_chartgalaxy_result.json", "w"), indent=1)
