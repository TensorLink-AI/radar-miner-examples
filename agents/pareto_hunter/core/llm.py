"""LLM client — uses GatedClient to call the validator-provided LLM endpoint."""

import sys


DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3-0324-TEE"
MAX_RETRIES = 3


def get_models(client, llm_url: str) -> list[str]:
    """List available models from the LLM endpoint."""
    try:
        resp = client.get_json(f"{llm_url}/models")
        if isinstance(resp, list):
            return resp
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

    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.post_json(f"{llm_url}/chat", payload)
            return resp["content"]
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
