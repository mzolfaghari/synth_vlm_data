"""Pluggable LLM backend for the COMPACT pipeline.

COMPACT (github.com/princetonvisualai/compact) calls Gemini via the `google.generativeai`
module, which its generator.py / verifier.py use as `client.GenerativeModel(name)
.generate_content(contents=[...], generation_config={...}).text`.

To run the SAME pipeline on our self-hosted Qwen (served by vLLM behind an OpenAI-compatible
endpoint) we provide `OpenAICompatClient`, a drop-in that mimics exactly that interface. This way
generator.py and verifier.py are used verbatim from upstream — only the `client` object differs.

Choose the backend with `make_client(...)`:
  * backend="openai" -> OpenAICompatClient (our Qwen vLLM endpoint; no API key needed)
  * backend="gemini" -> the upstream google.generativeai module (needs GEMINI api key)
"""
import os, io, time, base64
import requests

MAX_SIDE = int(os.environ.get("COMPACT_MAX_SIDE", "1280"))  # downscale long side sent to the VLM


def _encode_image(image_bytes):
    """Bytes -> base64 JPEG string, long side capped at MAX_SIDE (keeps text legible, bounds tokens)."""
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = im.size
        m = max(w, h)
        if m > MAX_SIDE:
            s = MAX_SIDE / m
            im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        # PIL missing / already-encoded: pass bytes through as base64
        return base64.b64encode(image_bytes).decode()


class _Response:
    """Mimics google.generativeai's response object: only `.text` is used upstream."""
    def __init__(self, text):
        self.text = text


class _OpenAICompatModel:
    def __init__(self, base_url, model, timeout=300, retries=3, enable_thinking=False):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.enable_thinking = enable_thinking

    def generate_content(self, contents, generation_config=None):
        """Map Gemini-style `contents` (list of str and {"mime_type","data":bytes} parts) and
        `generation_config` onto an OpenAI /chat/completions call. All parts become one user
        message (Gemini has no separate system role; upstream folds everything into contents)."""
        cfg = generation_config or {}
        text_parts, parts = [], []
        for c in contents:
            if isinstance(c, str):
                text_parts.append(c)
            elif isinstance(c, dict) and "data" in c:
                b64 = _encode_image(c["data"])
                parts.append({"type": "image_url",
                              "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        parts.append({"type": "text", "text": "\n\n".join(text_parts)})
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": parts}],
            "temperature": cfg.get("temperature", 0.1),
            "top_p": cfg.get("top_p", 0.9),
            "max_tokens": cfg.get("max_output_tokens", 1000),
            # Qwen3 is a reasoning model; COMPACT's Q&A is perception, not multi-step reasoning
            # — disable "thinking" so calls are fast and return plain JSON (as chartgalaxy does).
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        last = None
        for attempt in range(self.retries):
            try:
                r = requests.post(f"{self.base_url}/chat/completions", json=payload,
                                  timeout=self.timeout)
                r.raise_for_status()
                return _Response(r.json()["choices"][0]["message"]["content"])
            except Exception as e:
                last = e
                time.sleep(2 * (attempt + 1))
        raise last


class OpenAICompatClient:
    """Drop-in for the `google.generativeai` module used by COMPACT's generator/verifier."""
    def __init__(self, base_url, model):
        self.base_url = base_url
        self.model = model

    def configure(self, **kwargs):  # no-op; parity with genai.configure(api_key=...)
        pass

    def GenerativeModel(self, name=None):  # name (e.g. 'models/gemini-2.0-flash') is ignored
        return _OpenAICompatModel(self.base_url, self.model)


def make_client(backend="openai", api_key=None,
                base_url=None, model=None):
    """Build the `client` passed to generate_questions()/verify_question_capabilities()."""
    if backend == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai
    if backend == "openai":
        base_url = base_url or os.environ.get("QA_LLM_BASE_URL", "http://localhost:8000/v1")
        model = model or os.environ.get("QA_LLM_MODEL", "Qwen/Qwen3.6-27B-FP8")
        return OpenAICompatClient(base_url, model)
    raise ValueError(f"unknown backend: {backend}")
