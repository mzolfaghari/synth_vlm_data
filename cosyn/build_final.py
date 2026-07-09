#!/usr/bin/env python3
"""Build KUB2M_3Bsynth_MathEnrich_OCRDoc_ChartCoSyn.jsonl:
  OCRDoc base                       (KUB2M_3Bsynth_MathEnrich_OCRDoc.jsonl, 3,386,488)
  + chart_doc_enrich/chartqa.jsonl       (ChartQA base QA)
  + chart_doc_enrich/chart2text.jsonl    (Statista/Pew chart descriptions)
  + cosyn_enrich/cosyn_*.jsonl           (7 CoSyn-400K configs, ~300k, decontaminated)
All rows normalized to {config, image, question, answer}.
"""
import json, glob, os

BASE = "/home/reza/data/KUB2M_3Bsynth_MathEnrich_OCRDoc.jsonl"
CHART_DOC = ["/home/reza/data/chart_doc_enrich/chartqa.jsonl",
             "/home/reza/data/chart_doc_enrich/chart2text.jsonl"]
COSYN = sorted(glob.glob("/home/reza/data/cosyn_enrich/cosyn_*.jsonl"))
OUT = "/home/reza/data/KUB2M_3Bsynth_MathEnrich_OCRDoc_ChartCoSyn.jsonl"

KEYS = ("config", "image", "question", "answer")
def norm(o):
    return json.dumps({k: o.get(k) for k in KEYS}, ensure_ascii=False)

n = 0; per = {}
with open(OUT, "w") as g:
    # OCRDoc base (pass-through verbatim)
    c = 0
    for line in open(BASE):
        g.write(line if line.endswith("\n") else line + "\n"); c += 1; n += 1
    per["ocr_doc_base"] = c
    print(f"base: {c:,}", flush=True)

    # chart_doc slices
    for path in CHART_DOC:
        c = 0
        for ln in open(path):
            g.write(norm(json.loads(ln)) + "\n"); c += 1; n += 1
        per[os.path.basename(path).replace(".jsonl", "")] = c
        print(f"{os.path.basename(path)}: {c:,}", flush=True)

    # CoSyn configs
    for path in COSYN:
        c = 0
        for ln in open(path):
            g.write(norm(json.loads(ln)) + "\n"); c += 1; n += 1
        per[os.path.basename(path).replace(".jsonl", "")] = c
        print(f"{os.path.basename(path)}: {c:,}", flush=True)

print(f"\nTOTAL: {n:,}")
print("per-source:", json.dumps(per, indent=1))
json.dump({"total": n, "per_source": per, "output": OUT},
          open("/home/reza/data/cosyn_enrich/build_final_result.json", "w"), indent=1)
