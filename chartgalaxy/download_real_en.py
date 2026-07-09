#!/usr/bin/env python3
"""ChartGalaxy pipeline — STAGE 1: fetch REAL infographics, keep ENGLISH only.

Real ChartGalaxy ships URL-only images (Statista / Visual Capitalist / Pinterest) + a tabular
`data` field. We: stream the real shards, language-detect each chart's title/label text
(langdetect, en prob > EN_PROB), download the image from its URL, and write a metadata record
per kept image:
    {id, config, image (absolute local path), chart_type, source, title, data}
to real_en_meta.jsonl  (consumed by gen_qa.py).

Real ChartGalaxy is ~60-72% English, so most charts survive the filter; non-English (mostly
German/French/Spanish) and number-only tables are dropped.

Usage:
  download_real_en.py --limit 20000 [--shards 0 1 2 ...]
Env: none. Uses only /home/reza paths.
"""
import os, io, json, sys, time, argparse, urllib.request
import tarfile
from huggingface_hub import hf_hub_download, list_repo_files
from langdetect import detect_langs, DetectorFactory
from PIL import Image
DetectorFactory.seed = 0
Image.MAX_IMAGE_PIXELS = None

ROOT = "/home/reza/data/chartgalaxy_real_en"
IMG_DIR = f"{ROOT}/images"
META = f"{ROOT}/real_en_meta.jsonl"
EN_PROB = 0.90
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120 Safari/537.36"}

def texts_from_data(d):
    out = []
    def walk(x):
        if isinstance(x, str): out.append(x)
        elif isinstance(x, list):
            for e in x: walk(e)
        elif isinstance(x, dict):
            for e in x.values(): walk(e)
    try: walk(json.loads(d) if isinstance(d, str) else d)
    except Exception: pass
    return [s for s in out if any(c.isalpha() for c in s)]

def is_english(data_field):
    txt = " ".join(texts_from_data(data_field))
    if len(txt) < 12: return False, txt, None
    try:
        top = detect_langs(txt)[0]
        return (top.lang == "en" and top.prob > EN_PROB), txt, f"{top.lang}:{top.prob:.2f}"
    except Exception:
        return False, txt, None

def fetch_image(url):
    try:
        data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=15).read()
        im = Image.open(io.BytesIO(data)).convert("RGB")
        return im if min(im.size) >= 150 else None
    except Exception:
        return None

def iter_real_entries(shard_ids):
    files = [f for f in list_repo_files("ChartGalaxy/ChartGalaxy", repo_type="dataset")
             if f.startswith("real/dataset_") and f.endswith(".tar")]
    files.sort()
    if shard_ids:
        files = [f for f in files if any(f"dataset_{i:03d}." in f for i in shard_ids)]
    for f in files:
        p = hf_hub_download("ChartGalaxy/ChartGalaxy", f, repo_type="dataset", local_dir="/tmp/cg_real")
        with tarfile.open(p) as t:
            for m in t.getmembers():
                if m.name.endswith(".json"):
                    try: yield json.loads(t.extractfile(m).read().decode("utf-8", "replace"))
                    except Exception: continue

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20000, help="max English images to keep")
    ap.add_argument("--shards", type=int, nargs="*", default=None, help="real shard ids (default: all)")
    args = ap.parse_args()

    os.makedirs(IMG_DIR, exist_ok=True)
    done_ids = set()
    if os.path.exists(META):
        for ln in open(META):
            try: done_ids.add(json.loads(ln)["id"])
            except Exception: pass
    out = open(META, "a")
    kept = len(done_ids); seen = en_txt = fetched = 0
    t0 = time.time()
    for o in iter_real_entries(args.shards):
        if kept >= args.limit: break
        cid = o.get("id"); url = o.get("image_url") or o.get("url")
        if not cid or not url or cid in done_ids: continue
        seen += 1
        ok, title, lang = is_english(o.get("data"))
        if not ok: continue
        en_txt += 1
        im = fetch_image(url)
        if im is None: continue
        fetched += 1
        ipath = f"{IMG_DIR}/{cid}.jpg"
        try: im.save(ipath, "JPEG", quality=92)
        except Exception: continue
        out.write(json.dumps({"id": cid, "config": "chartgalaxy_real_en", "image": ipath,
                              "chart_type": o.get("chart_type") or o.get("type"),
                              "source": o.get("source"), "lang": lang,
                              "title": title[:200], "data": o.get("data")},
                             ensure_ascii=False) + "\n")
        out.flush(); done_ids.add(cid); kept += 1
        if kept % 200 == 0:
            sys.stderr.write(f"kept={kept} seen={seen} en={en_txt} fetched={fetched} "
                             f"({kept/max(time.time()-t0,1):.1f}/s)\n"); sys.stderr.flush()
    out.close()
    print(f"DONE kept={kept} (target {args.limit}); scanned={seen}, english={en_txt}, "
          f"fetched_ok={fetched}. meta -> {META}")

if __name__ == "__main__":
    main()
