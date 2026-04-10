"""LLM client — uses GatedClient to call the validator-provided LLM endpoint."""

import json
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
    """
    if not llm_url:
        raise RuntimeError("No llm_url provided in challenge")

    url = f"{llm_url}/v1/chat/completions"

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
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.post_json(url, payload)
                break
            except Exception as exc:
                last_err = exc
                print(
                    f"[llm] round {round_idx + 1} attempt {attempt + 1}/{MAX_RETRIES} "
                    f"failed: {exc}",
                    file=sys.stderr,
                )

        if resp is None:
            raise RuntimeError(
                f"LLM call failed after {MAX_RETRIES} attempts: {last_err}"
            )

        choice = resp["choices"][0]
        assistant_msg = choice["message"]
        finish_reason = choice.get("finish_reason", "")

        tool_calls = assistant_msg.get("tool_calls")

        # --- No tool calls → return text content ---
        # NOTE: we intentionally do NOT short-circuit on
        # finish_reason == "stop". Some OpenAI-compatible servers (notably
        # certain Kimi deployments) return finish_reason="stop" together
        # with a populated tool_calls list; branching on finish_reason
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
        # Append the full assistant message (with tool_calls) to history.
        messages.append(assistant_msg)

        for tc in tool_calls:
            func = tc["function"]
            fn_name = func["name"]
            call_id = tc["id"]

            # Parse arguments
            try:
                kwargs = json.loads(func.get("arguments") or "{}")
            except json.JSONDecodeError:
                kwargs = {}

            # Dispatch to handler
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

    # All rounds exhausted — return whatever content the last response had.
    return assistant_msg.get("content") or ""


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
    # Fallback: accept unfenced code if it looks like valid model code
    if "def build_model" in text and "import" in text:
        return text.strip()
    # No code block found — return empty so validation rejects cleanly
    return ""
