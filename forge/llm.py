"""Minimal OpenAI-compatible chat helper for FORGE (self-hosted Qwen via vLLM).

Same house style as chartgalaxy/gen_qa.py: base64 image in the image_url slot, Qwen "thinking"
disabled, small retry loop. Kept dependency-light so the debate/gates modules import cleanly.
"""
import os, io, re, json, time, base64
import requests

MAX_SIDE = int(os.environ.get("FORGE_MAX_SIDE", "1280"))  # downscale long side sent to the VLM


def b64_image(path):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path).convert("RGB")
    w, h = im.size
    m = max(w, h)
    if m > MAX_SIDE:
        s = MAX_SIDE / m
        im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def chat(base_url, model, system, user_text, img_b64=None,
         temperature=0.1, max_tokens=800, timeout=300, retries=3):
    content = []
    if img_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
    content.append({"type": "text", "text": user_text})
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(f"{base_url.rstrip('/')}/chat/completions",
                              json={"model": model, "messages": msgs, "temperature": temperature,
                                    "max_tokens": max_tokens,
                                    "chat_template_kwargs": {"enable_thinking": False}},
                              timeout=timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


def parse_json(txt):
    if not txt:
        return None
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None
