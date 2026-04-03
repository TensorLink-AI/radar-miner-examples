"""Chutes LLM client — OpenAI-compatible chat completions via stdlib only."""

import json
import os
import sys
import time
import urllib.request
import urllib.error

BASE_URL = "https://llm.chutes.ai/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3-0324"
TIMEOUT = 120
MAX_RETRIES = 3


def chat(messages: list[dict], *, temperature: float = 0.7,
         max_tokens: int = 8192, model: str | None = None) -> str:
    """Send chat completion request and return assistant content."""
    model = model or os.environ.get("CHUTES_MODEL", DEFAULT_MODEL)
    api_key = os.environ.get("CHUTES_API_KEY", "")
    if not api_key:
        raise RuntimeError("CHUTES_API_KEY not set")

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            time.sleep(2 ** attempt)
        try:
            req = urllib.request.Request(
                f"{BASE_URL}/chat/completions",
                data=payload,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                body = json.loads(resp.read().decode())
            return body["choices"][0]["message"]["content"]
        except (urllib.error.URLError, urllib.error.HTTPError,
                json.JSONDecodeError, KeyError, TimeoutError) as exc:
            last_err = exc
            print(f"[llm] attempt {attempt + 1}/{MAX_RETRIES} failed: {exc}",
                  file=sys.stderr)

    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} attempts: {last_err}")


def extract_code(text: str) -> str:
    """Extract the first fenced python code block from LLM output."""
    markers = ["```python", "```Python", "```py"]
    for marker in markers:
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.index("```", start)
            return text[start:end].strip()
    # Fallback: if no fenced block, try to find bare triple-backtick block
    if "```" in text:
        start = text.index("```") + 3
        # Skip optional language tag on same line
        nl = text.index("\n", start)
        start = nl + 1
        end = text.index("```", start)
        return text[start:end].strip()
    # Last resort: return the whole text
    return text.strip()
