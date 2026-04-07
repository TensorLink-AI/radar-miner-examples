"""LLM client — uses GatedClient to call the validator-provided LLM endpoint."""

import sys


DEFAULT_MODEL = "moonshotai/Kimi-K2.5-TEE"
MAX_RETRIES = 3


def get_models(client, llm_url: str) -> list[str]:
    """List available models from the LLM endpoint."""
    try:
        resp = client.get_json(f"{llm_url}/v1/models")
        if isinstance(resp, list):
            return resp
        # OpenAI format: {"object": "list", "data": [{"id": "model-name", ...}]}
        if "data" in resp:
            return [m["id"] for m in resp["data"]]
        return resp.get("models", [])
    except Exception as exc:
        print(f"[llm] failed to list models: {exc}", file=sys.stderr)
        return []


def chat(client, llm_url: str, messages: list[dict], *,
         temperature: float = 0.7, max_tokens: int = 4096,
         model: str = DEFAULT_MODEL) -> str:
    """Send chat request via GatedClient and return assistant content."""
    if not llm_url:
        raise RuntimeError("No llm_url provided in challenge")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    print(f"[llm] calling {llm_url}/v1/chat/completions model={model} "
          f"msgs={len(messages)} temp={temperature}", file=sys.stderr)

    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.post_json(f"{llm_url}/v1/chat/completions", payload)
            content = resp["choices"][0]["message"]["content"]
            print(f"[llm] response received: {len(content)} chars", file=sys.stderr)
            return content
        except Exception as exc:
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
            closing = text.find("```", start)
            if closing == -1:
                # Truncated output — take everything after the marker
                return text[start:].strip()
            return text[start:closing].strip()
    # Fallback: if no fenced block, try to find bare triple-backtick block
    if "```" in text:
        start = text.index("```") + 3
        nl = text.find("\n", start)
        if nl == -1:
            return text[start:].strip()
        start = nl + 1
        closing = text.find("```", start)
        if closing == -1:
            return text[start:].strip()
        return text[start:closing].strip()
    # Fallback: if no fenced block but text looks like Python code, use it directly
    if "def build_model" in text and "import" in text:
        return text.strip()
    # No code block found — return empty so validation rejects cleanly
    return ""
