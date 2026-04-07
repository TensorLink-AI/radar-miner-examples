"""LLM reasoning engine — uses GatedClient to call the validator-provided LLM.

Key design decisions:
  - Up to 15 LLM calls per round (uses half the 30-request rate limit,
    leaving headroom for future enhancements)
  - Multi-turn conversation: each retry feeds back validation errors so the
    LLM can self-correct across turns
  - No internal retries within chat() — caller manages retry logic
  - Graceful degradation: returns None on failure, never raises
"""

import sys

from core.validation import validate_code
from core.prompt_builder import build_system_prompt, build_user_prompt

DEFAULT_MODEL = "moonshotai/Kimi-K2.5-TEE"
MAX_LLM_ATTEMPTS = 15  # up to 15 turns — half the 30-request rate limit


def chat(client, llm_url: str, messages: list[dict], *,
         temperature: float = 0.7, max_tokens: int = 4096,
         model: str = DEFAULT_MODEL) -> str:
    """Single LLM call. No internal retries — caller manages retry logic."""
    if not llm_url:
        raise RuntimeError("No llm_url provided")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    print(f"[llm] calling {llm_url}/v1/chat/completions model={model} "
          f"msgs={len(messages)} temp={temperature}", file=sys.stderr)

    resp = client.post_json(f"{llm_url}/v1/chat/completions", payload)
    content = resp["choices"][0]["message"]["content"]
    print(f"[llm] response received: {len(content)} chars", file=sys.stderr)
    return content


def extract_code(text: str) -> str:
    """Extract the first fenced python code block from LLM output."""
    markers = ["```python", "```Python", "```py"]
    for marker in markers:
        if marker in text:
            start = text.index(marker) + len(marker)
            closing = text.find("```", start)
            if closing == -1:
                return text[start:].strip()
            return text[start:closing].strip()
    # Fallback: bare triple-backtick block
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
    # Fallback: raw Python that looks like valid code
    if "def build_model" in text and "import" in text:
        return text.strip()
    return ""


def reason_and_generate(client, challenge: dict,
                        context: dict) -> tuple[str, str, str] | None:
    """Use the LLM to reason about architecture design and generate code.

    Returns (code, name, motivation) if LLM produced valid code, else None.
    Uses up to 15 LLM calls with multi-turn error correction. Each failed
    attempt feeds validation errors back to the LLM so it can self-correct.
    Returns as soon as valid code is produced.
    """
    llm_url = challenge.get("llm_url", "")
    if not llm_url:
        return None

    system_prompt = build_system_prompt(challenge)
    user_prompt = build_user_prompt(challenge, context)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    code = ""
    name = "reliable_baseline_submission"
    motivation = "LLM-designed architecture"

    for attempt in range(MAX_LLM_ATTEMPTS):
        print(f"[llm] attempt {attempt + 1}/{MAX_LLM_ATTEMPTS}",
              file=sys.stderr)

        try:
            response = chat(client, llm_url, messages, temperature=0.7)
            code = extract_code(response)

            # Try to extract name/motivation from response
            for line in response.split("\n"):
                stripped = line.strip()
                if stripped.startswith("# Name:"):
                    name = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("# Motivation:"):
                    motivation = stripped.split(":", 1)[1].strip()

            ok, errors = validate_code(code)
            if ok:
                print(f"[llm] validation passed on attempt {attempt + 1}",
                      file=sys.stderr)
                return code, name, motivation

            # Validation failed — add error feedback for retry
            print(f"[llm] validation errors: {errors}", file=sys.stderr)
            if attempt < MAX_LLM_ATTEMPTS - 1:
                if code:
                    messages.append(
                        {"role": "assistant", "content": f"```python\n{code}\n```"})
                    messages.append({"role": "user", "content": (
                        "The code has validation errors:\n"
                        + "\n".join(f"- {e}" for e in errors)
                        + "\n\nFix these errors and return the corrected code. "
                        "Remember: build_model and build_optimizer MUST be "
                        "top-level def statements (not inside a class)."
                    )})
                else:
                    messages.append({"role": "user", "content": (
                        "Previous attempt failed: no Python code block found. "
                        "You MUST respond with a single ```python code block "
                        "containing def build_model(context_len, prediction_len, "
                        "num_variates, quantiles) and def build_optimizer(model)."
                    )})

        except Exception as exc:
            print(f"[llm] attempt {attempt + 1} error: {exc}", file=sys.stderr)
            if attempt < MAX_LLM_ATTEMPTS - 1:
                messages.append({"role": "user", "content": (
                    "Previous request failed. Please try again with a single "
                    "```python code block containing build_model and "
                    "build_optimizer functions."
                )})

    print("[llm] all attempts exhausted, returning None", file=sys.stderr)
    return None
