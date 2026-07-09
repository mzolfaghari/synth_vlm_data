#!/usr/bin/env python3
"""Convert COMPACT output (compact/main.py -> output_k{k}.json) into our training-manifest
schema {config, image, question, answer}, one JSON row per QA turn — so COMPACT data drops
straight into finetune_mixed.py --mixed like the chartgalaxy / cosyn manifests.

COMPACT writes each image as a pretty-printed JSON array `[entry]` (entry = LLaVA-style
{id, image, conversations:[{from:human,value},{from:gpt,value},...]}), concatenated with
newlines — not line-delimited JSON. We stream-decode those concatenated values.

The first human turn carries the LLaVA prefix/suffix ("<image>\\n...\\nAnswer the question using
a single word or phrase."); we strip both so the manifest holds the bare question.

Usage:
  python -m compact.to_manifest --inputs output/output_k1.json output/output_k2.json output/output_k3.json \\
         --image-root /home/reza/data/cosyn_images --config compact_cosyn --out /home/reza/data/compact_cosyn.jsonl
"""
import os, re, json, argparse

_SUFFIX = re.compile(r"\s*Answer the question using a single word or phrase\.\s*$", re.I)


def read_entries(path):
    """Yield entries from COMPACT's concatenated `[entry]` JSON blocks."""
    txt = open(path).read()
    dec = json.JSONDecoder()
    i, n = 0, len(txt)
    while i < n:
        while i < n and txt[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        obj, j = dec.raw_decode(txt, i)
        i = j
        for e in (obj if isinstance(obj, list) else [obj]):
            yield e


def clean_question(v):
    v = v.strip()
    if v.startswith("<image>"):
        v = v[len("<image>"):].lstrip("\n").strip()
    v = _SUFFIX.sub("", v).strip()
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="COMPACT output_k*.json files")
    ap.add_argument("--out", required=True, help="manifest JSONL to write")
    ap.add_argument("--image-root", default="", help="prepended to each entry's image (make absolute for --mixed)")
    ap.add_argument("--config", default="compact", help="config label written to every row")
    ap.add_argument("--keep-capability", action="store_true", help="also emit the turn's capability list + k")
    args = ap.parse_args()

    n_img = n_rows = 0
    with open(args.out, "w") as g:
        for path in args.inputs:
            for e in read_entries(path):
                img = e.get("image", "")
                if args.image_root:
                    img = os.path.join(args.image_root, img)
                conv = e.get("conversations", [])
                n_img += 1
                for t in range(0, len(conv) - 1, 2):
                    hum, gpt = conv[t], conv[t + 1]
                    if hum.get("from") != "human" or gpt.get("from") != "gpt":
                        continue
                    q = clean_question(str(hum.get("value", "")))
                    a = str(gpt.get("value", "")).strip()
                    if not q or not a:
                        continue
                    row = {"config": args.config, "image": img, "question": q, "answer": a}
                    if args.keep_capability:
                        row["capability"] = hum.get("capability")
                        row["k"] = len(hum.get("capability") or []) or None
                    g.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_rows += 1
    print(json.dumps({"inputs": args.inputs, "images": n_img, "rows": n_rows, "out": args.out}, indent=1))


if __name__ == "__main__":
    main()
