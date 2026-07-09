#!/usr/bin/env python3
"""Download CoSyn-400K (allenai) text-rich synthetic data, decontaminate vs the 18
eval benchmarks (reuse math_sft_enrich/decontam/eval_hashes.json), emit SFT rows
{config, image (absolute path), question, answer} with ABSOLUTE image paths.

CoSyn schema: row = {id, image(PIL), qa_pairs{question[],explanation[],answer[]},
metadata, data, code}. Single image per row. We keep up to QA_PER_IMAGE pairs/image.

Configs chosen to target our measured weaknesses:
  document/table/nutrition -> OCRBench KIE regression + MMVet ocr
  chart/diagram/graphic    -> InfoVQA
  math                     -> MMVet ocr_math
Skipped (irrelevant to our 18 benchmarks): chemical, circuit, music.

Usage: download.py <plan_key>   plan_key in the per-config labels or 'all'.
Images are resized so the long side <= MAX_SIDE to bound disk + keep text legible.
"""
import os, json, sys, time
from datasets import load_dataset
from PIL import Image
import imagehash
Image.MAX_IMAGE_PIXELS = None

ROOT = "/home/reza/data/cosyn_enrich"
EVAL = json.load(open("/home/reza/data/math_sft_enrich/decontam/eval_hashes.json"))
EPH = set(EVAL["phash"]); EDH = set(EVAL["dhash"])

QA_PER_IMAGE = 3
MAX_SIDE = 1536

# label -> (hf_config, qa_row_target)  -- targets sum to ~300k rows
PLANS = {
    "cosyn_document":  ("document",  70_000),  # KIE + ocr
    "cosyn_table":     ("table",     45_000),  # KIE + ocr_math
    "cosyn_chart":     ("chart",     60_000),  # InfoVQA
    "cosyn_math":      ("math",      40_000),  # ocr_math
    "cosyn_diagram":   ("diagram",   35_000),  # InfoVQA / AI2D
    "cosyn_graphic":   ("graphic",   30_000),  # InfoVQA
    "cosyn_nutrition": ("nutrition", 20_000),  # KIE (all ~7k imgs x 3)
}

def hashes(im):
    g = im.convert("L")
    return str(imagehash.phash(g, 8)), str(imagehash.dhash(g, 8))

def resize_long(im, max_side=MAX_SIDE):
    w, h = im.size
    m = max(w, h)
    if m <= max_side:
        return im
    s = max_side / m
    return im.resize((max(1, int(w*s)), max(1, int(h*s))), Image.LANCZOS)

def run(plan_key):
    os.makedirs(ROOT, exist_ok=True)
    labels = list(PLANS) if plan_key == "all" else [plan_key]
    for label in labels:
        cfg, target = PLANS[label]
        out_jsonl = open(f"{ROOT}/{label}.jsonl", "w")
        saved_hashes = {}
        imgdir = f"{ROOT}/images/{label}"
        os.makedirs(imgdir, exist_ok=True)
        ds = load_dataset("allenai/CoSyn-400K", cfg, split="train", streaming=True)
        qa = imgs = contam = seen = noqa = 0
        img_counter = 0
        t0 = time.time()
        for row in ds:
            seen += 1
            im = row.get("image")
            if im is None:
                continue
            try:
                p, d = hashes(im)
            except Exception:
                continue
            if p in EPH or d in EDH:
                contam += 1
                if qa >= target: break
                continue
            qap = row.get("qa_pairs") or {}
            qs = qap.get("question") or []
            ans = qap.get("answer") or []
            pairs = [(q, a) for q, a in zip(qs, ans)
                     if isinstance(q, str) and q.strip()
                     and isinstance(a, str) and a.strip()][:QA_PER_IMAGE]
            if not pairs:
                noqa += 1
                continue
            ipath = f"{imgdir}/{img_counter:08d}.jpg"
            try:
                resize_long(im).convert("RGB").save(ipath, "JPEG", quality=90)
            except Exception:
                continue
            img_counter += 1; imgs += 1
            saved_hashes[ipath] = p + ":" + d
            for q, a in pairs:
                out_jsonl.write(json.dumps(
                    {"config": label, "image": ipath, "question": q, "answer": a},
                    ensure_ascii=False) + "\n")
                qa += 1
            if imgs % 2000 == 0:
                sys.stderr.write(
                    f"[{label}] seen={seen} imgs={imgs} qa={qa} contam={contam} "
                    f"noqa={noqa} ({qa/max(time.time()-t0,1):.0f} qa/s)\n")
                sys.stderr.flush()
            if qa >= target:
                break
        out_jsonl.close()
        stats = dict(config=cfg, seen=seen, images=imgs, qa_rows=qa,
                     contaminated=contam, no_qa=noqa, target=target)
        json.dump(saved_hashes, open(f"{ROOT}/{label}_image_hashes.json", "w"))
        json.dump(stats, open(f"{ROOT}/{label}_stats.json", "w"), indent=1)
        sys.stderr.write(f"DONE {label}: {json.dumps(stats)}\n"); sys.stderr.flush()
        print(f"DONE {label}: {json.dumps(stats)}", flush=True)

if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "all")
