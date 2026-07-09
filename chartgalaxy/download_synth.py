#!/usr/bin/env python3
"""ChartGalaxy pipeline — STAGE 1 (SYNTHETIC): pull bundled synthetic infographics.

Unlike the real split, synthetic shards (synthetic/dataset_XXX.tar, 366 shards x 5000 imgs) ship
the PNG image + `.info.json` (chart_type/layout) + `.data.json` (table) bundled — so no URL
scraping and no language attrition (all English). We extract images + build a meta file identical
in shape to the real one, consumed by gen_qa.py.

Meta row: {id, config:"chartgalaxy_synth", image (abs path), chart_type, data}
Resumable (skips ids already in the meta). Downloaded tars are deleted after extraction.

Usage: download_synth.py --limit 200000
"""
import os, io, json, sys, time, tarfile, argparse
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ROOT = "/home/reza/data/chartgalaxy_synth"
IMG_DIR = f"{ROOT}/images"
META = f"{ROOT}/synth_meta.jsonl"

def run(target):
    os.makedirs(IMG_DIR, exist_ok=True)
    shards = sorted(f for f in list_repo_files("ChartGalaxy/ChartGalaxy", repo_type="dataset")
                    if f.startswith("synthetic/dataset_") and f.endswith(".tar"))
    done = set()
    if os.path.exists(META):
        for l in open(META):
            try: done.add(json.loads(l)["id"])
            except Exception: pass
    out = open(META, "a"); kept = len(done); t0 = time.time()
    for f in shards:
        if kept >= target: break
        shard = f.split("_")[-1].split(".")[0]                 # e.g. "012"
        p = hf_hub_download("ChartGalaxy/ChartGalaxy", f, repo_type="dataset", local_dir="/tmp/cg_synth")
        with tarfile.open(p) as t:
            members = {m.name: m for m in t.getmembers()}
            for name in members:
                if kept >= target: break
                if not name.endswith(".png"): continue
                stem = name[:-4]                                # "./00000000"
                cid = f"{shard}_{stem.lstrip('./')}"            # unique across shards
                if cid in done: continue
                dj, ij = stem + ".data.json", stem + ".info.json"
                if dj not in members: continue                 # need a table for generation
                try:
                    data = json.loads(t.extractfile(members[dj]).read().decode("utf-8", "replace"))
                except Exception:
                    continue
                info = {}
                if ij in members:
                    try: info = json.loads(t.extractfile(members[ij]).read().decode("utf-8", "replace"))
                    except Exception: pass
                try:
                    im = Image.open(t.extractfile(members[name])).convert("RGB")
                except Exception:
                    continue
                ipath = f"{IMG_DIR}/{cid}.jpg"
                try: im.save(ipath, "JPEG", quality=90)
                except Exception: continue
                out.write(json.dumps({"id": cid, "config": "chartgalaxy_synth", "image": ipath,
                                      "chart_type": info.get("chart_type") or info.get("chart_variation"),
                                      "data": data}, ensure_ascii=False) + "\n")
                out.flush(); done.add(cid); kept += 1
                if kept % 2000 == 0:
                    sys.stderr.write(f"kept={kept} shard={shard} ({kept/max(time.time()-t0,1):.0f}/s)\n")
                    sys.stderr.flush()
        try: os.remove(p)                                       # free the ~500MB tar
        except OSError: pass
    out.close()
    print(f"DONE kept={kept} (target {target}) -> {META}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200000)
    run(ap.parse_args().limit)
