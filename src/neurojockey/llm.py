"""Client for any OpenAI-compatible chat completions API (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL)."""

import json
import os
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "https://api.openai.com/v1"


def load_env(path: Path = Path(".env")) -> None:
    """Minimal .env loader (KEY=VALUE lines); existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def model_name(model: str | None = None) -> str:
    load_env()
    model = model or os.environ.get("LLM_MODEL")
    if not model:
        raise RuntimeError("LLM_MODEL is not set (put it in .env or pass --llm-model)")
    return model


def ask(model: str, system: str, payload: dict, schema: dict, name: str) -> dict:
    """One chat completion with a strict JSON-schema answer, parsed."""
    load_env()
    key = os.environ.get("LLM_API_KEY")
    if not key:
        raise RuntimeError("LLM_API_KEY is not set (put it in .env)")
    url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}},
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.loads(r.read())
    if "error" in resp:
        raise RuntimeError(resp["error"].get("message", resp["error"]))
    return json.loads(resp["choices"][0]["message"]["content"])
