"""LLM reasoning engine — uses GatedClient to call the validator-provided LLM.

Key design decisions:
  - Up to 15 LLM calls per round (uses half the 30-request rate limit,
    leaving headroom for future enhancements)
  - Multi-turn conversation: each retry feeds back validation errors so the
    LLM can self-correct across turns
  - No internal retries within chat() — caller manages retry logic
  - Graceful degradation: returns None on failure, never raises
"""

import json
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


def chat_with_tools(
    client,
    llm_url: str,
    messages: list[dict],
    tools: list[dict],
    tool_handlers: dict,
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    max_rounds: int = 8,
    model: str = DEFAULT_MODEL,
) -> str:
    """Multi-round tool-calling loop using the OpenAI chat completions format.

    Sends ``tools`` definitions alongside messages. When the assistant responds
    with ``tool_calls``, each call is dispatched to the matching handler in
    *tool_handlers*, and the results are appended as ``role: tool`` messages
    before the next request.  The loop continues until the model returns a plain
    text response (``finish_reason == "stop"``) or *max_rounds* is exhausted.

    Unlike ``chat()``, this function retries transient HTTP failures internally
    (up to 3 attempts per round) so callers don't need their own retry wrapper.
    """
    if not llm_url:
        raise RuntimeError("No llm_url provided")

    url = f"{llm_url}/v1/chat/completions"
    max_retries = 3

    for round_idx in range(max_rounds):
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # On the final round, omit tools to force a text response.
        if round_idx < max_rounds - 1 and tools:
            payload["tools"] = tools

        print(
            f"[llm] tool-call round {round_idx + 1}/{max_rounds} "
            f"msgs={len(messages)} temp={temperature}",
            file=sys.stderr,
        )

        # Retry loop for transient HTTP failures within a single round.
        last_err: Exception | None = None
        resp = None
        for attempt in range(max_retries):
            try:
                resp = client.post_json(url, payload)
                break
            except Exception as exc:
                last_err = exc
                print(
                    f"[llm] round {round_idx + 1} attempt {attempt + 1}/{max_retries} "
                    f"failed: {exc}",
                    file=sys.stderr,
                )

        if resp is None:
            raise RuntimeError(
                f"LLM call failed after {max_retries} attempts: {last_err}"
            )

        choice = resp["choices"][0]
        assistant_msg = choice["message"]
        finish_reason = choice.get("finish_reason", "")

        tool_calls = assistant_msg.get("tool_calls")

        # --- No tool calls → return text content ---
        # NOTE: we intentionally do NOT short-circuit on
        # ``finish_reason == "stop"``. Some OpenAI-compatible servers (notably
        # certain Kimi deployments) return ``finish_reason="stop"`` together
        # with a populated ``tool_calls`` list; branching on finish_reason
        # would silently drop those tool calls.
        if not tool_calls:
            content = assistant_msg.get("content") or ""
            print(
                f"[llm] final response: {len(content)} chars "
                f"(finish_reason={finish_reason})",
                file=sys.stderr,
            )
            return content

        # --- Process tool calls ---
        messages.append(assistant_msg)

        for tc in tool_calls:
            func = tc["function"]
            fn_name = func["name"]
            call_id = tc["id"]

            try:
                kwargs = json.loads(func.get("arguments") or "{}")
            except json.JSONDecodeError:
                kwargs = {}

            handler = tool_handlers.get(fn_name)
            if handler is None:
                result_str = f"Tool error: unknown tool '{fn_name}'"
                print(f"[llm] unknown tool: {fn_name}", file=sys.stderr)
            else:
                try:
                    result = handler(**kwargs)
                    result_str = str(result)
                except Exception as exc:
                    result_str = f"Tool error: {exc}"
                    print(
                        f"[llm] tool '{fn_name}' raised: {exc}", file=sys.stderr
                    )

            print(
                f"[llm] tool {fn_name} → {len(result_str)} chars",
                file=sys.stderr,
            )
            messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": result_str}
            )

    # All rounds exhausted.
    return assistant_msg.get("content") or ""


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

            ok, errors = validate_code(code, challenge)
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
                    tp = challenge.get("task", {}).get("task_params", {})
                    param_str = ", ".join(tp.keys()) if tp else "**task_params"
                    messages.append({"role": "user", "content": (
                        "Previous attempt failed: no Python code block found. "
                        "You MUST respond with a single ```python code block "
                        f"containing def build_model({param_str}) and def build_optimizer(model)."
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
