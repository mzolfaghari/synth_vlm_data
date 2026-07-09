#!/usr/bin/env python3
"""Version C — ChartGalaxy-only manifest: synthetic + real(v2), no KUB base.
Rows normalized to {config,image,question,answer}."""
import json
SYNTH = "/home/reza/data/chartgalaxy_synth/synth_qa.jsonl"
REAL = "/home/reza/data/chartgalaxy_real_en/real_en_qa_v2.jsonl"
OUT = "/home/reza/data/ChartGalaxy_only.jsonl"
KEYS = ("config", "image", "question", "answer")
def norm(line):
    o = json.loads(line); return json.dumps({k: o.get(k) for k in KEYS}, ensure_ascii=False) + "\n"
ns = nr = 0
with open(OUT, "w") as g:
    for line in open(SYNTH):
        if line.strip(): g.write(norm(line)); ns += 1
    for line in open(REAL):
        if line.strip(): g.write(norm(line)); nr += 1
print(json.dumps({"synth": ns, "real_v2": nr, "total": ns + nr, "OUT": OUT}, indent=1))
