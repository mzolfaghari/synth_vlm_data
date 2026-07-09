#!/usr/bin/env python3
"""FORGE orchestrator — turn a QA set over rendered images into a fidelity-checked manifest.

For each image (rows grouped by image), in order:
  1. cheap deterministic grounding gates (gates.triage): accept / drop / adjudicate
  2. the ambiguous residual goes to the adversarial panel (--mode debate) or a single judge
     (--mode single, the A/B baseline)
Kept pairs are written as {config, image, question, answer} (+ tier if present), so the output
folds straight into finetune_mixed.py --mixed alongside the chartgalaxy / cosyn manifests.

Input JSONL rows: {image, question, answer, [<source-key>: structured source], [tier]}.
Resumable: processed image ids are appended to <out>.done and skipped on re-run.

Usage (against our served Qwen):
  QA_LLM_BASE_URL=http://<node>:<port>/v1 QA_LLM_MODEL=Qwen/Qwen3.6-27B-FP8 \
    python -m forge.forge_verify --in cosyn_qa.jsonl --image-root /home/reza/data/cosyn_images \
      --out /home/reza/data/forge_cosyn.jsonl --config forge_cosyn --mode debate --workers 32
"""
import os, sys, json, time, argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .llm import b64_image
from .gates import triage, source_numbers, norm_answer
from .debate import adjudicate, single_verify


def process_image(image, rows, args):
    """Run gates + panel over all QA of one image. Returns (kept_rows, stats)."""
    stats = defaultdict(int)
    path = image if os.path.isabs(image) else os.path.join(args.image_root, image)
    try:
        img_b64 = b64_image(path)
    except Exception:
        stats["img_error"] += 1
        return [], stats
    kept, kept_qs = [], []
    for r in rows:
        q = str(r.get("question", "")).strip()
        a = norm_answer(r.get("answer", ""))
        if not q or not a:
            stats["empty"] += 1
            continue
        src = r.get(args.source_key)
        snums = source_numbers(src) if src is not None else []
        context = None
        if src is not None and args.pass_source:
            context = json.dumps(src, ensure_ascii=False)[:3500]
        # cheap sanity gates only DROP junk; every survivor is adjudicated by the panel
        decision, reason = triage(q, a, kept_qs)
        stats[f"gate_{reason}"] += 1
        if decision == "drop":
            continue
        if args.mode == "single":
            v = single_verify(args.base_url, args.model, img_b64, q, a, context)
        else:
            v = adjudicate(args.base_url, args.model, img_b64, q, a, context, snums,
                           rounds=args.rounds, r_max=args.rmax)
        stats["panel_seen"] += 1
        if not v["kept"]:
            stats["panel_reject"] += 1
            continue
        stats["panel_accept"] += 1
        if v.get("refined"):
            stats["refined"] += 1
        final_a = v["answer"]
        row = {"config": args.config, "image": path, "question": q, "answer": final_a}
        if r.get("tier"):
            row["tier"] = r["tier"]
        kept.append(row)
        kept_qs.append(q)
    stats["kept"] += len(kept)
    return kept, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True, help="input QA jsonl")
    ap.add_argument("--out", required=True, help="output manifest jsonl (resumable)")
    ap.add_argument("--image-root", default="", help="prepended to non-absolute image paths")
    ap.add_argument("--config", default="forge", help="config label written to each row")
    ap.add_argument("--mode", choices=["debate", "single"], default="debate")
    ap.add_argument("--rounds", type=int, default=2, help="debate rounds T")
    ap.add_argument("--rmax", type=int, default=1, help="max dissent-driven refinements")
    ap.add_argument("--source-key", default="data", help="row field holding the structured source")
    ap.add_argument("--pass-source", action="store_true",
                    help="also give judges the structured source as context (grounding)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    ap.add_argument("--base-url", default=os.environ.get("QA_LLM_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("QA_LLM_MODEL", "Qwen/Qwen3.6-27B-FP8"))
    ap.add_argument("--report", default="", help="optional path to write a JSON stats report")
    args = ap.parse_args()

    groups = defaultdict(list)
    for line in open(args.infile):
        if line.strip():
            r = json.loads(line)
            groups[r["image"]].append(r)

    done_path = args.out + ".done"
    done = set()
    if os.path.exists(done_path):
        done = {l.strip() for l in open(done_path) if l.strip()}
    images = [im for im in groups if im not in done]
    if args.limit:
        images = images[:args.limit]
    print(f"images to process: {len(images)} (mode={args.mode}, endpoint={args.base_url}, out={args.out})",
          flush=True)

    out = open(args.out, "a")
    donef = open(done_path, "a")
    totals = defaultdict(int)
    n_img = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_image, im, groups[im], args): im for im in images}
        for fut in as_completed(futs):
            im = futs[fut]
            try:
                kept, stats = fut.result()
            except Exception as e:
                sys.stderr.write(f"error on {im}: {e}\n")
                continue
            for row in kept:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            donef.write(im + "\n"); donef.flush()
            for k, v in stats.items():
                totals[k] += v
            n_img += 1
            if n_img % 50 == 0:
                sys.stderr.write(f"imgs={n_img}/{len(images)} kept={totals['kept']} "
                                 f"panel={totals['panel_seen']} rej={totals['panel_reject']} "
                                 f"({n_img/max(time.time()-t0,1):.1f} img/s)\n")
                sys.stderr.flush()
    out.close(); donef.close()
    report = {"mode": args.mode, "images": n_img, **{k: totals[k] for k in sorted(totals)}}
    print(json.dumps(report, indent=1))
    if args.report:
        json.dump(report, open(args.report, "w"), indent=1)


if __name__ == "__main__":
    main()
