"""Reference miner agent that uses the OpenAI Python SDK directly.

Entry point: ``design_architecture(challenge, gated_client) -> dict``

The validator's ``GatedClient`` is intentionally ignored — egress is
already gated at the network layer (iptables) inside the agent pod, so
this agent talks to the OpenAI-compatible LLM proxy through the SDK and
keeps the rest of the agent code small and idiomatic.

Three phases:
  1. ``research`` — at most 20% of the budget (capped at 120s) — let the
     LLM call ``analyze_task`` / ``list_frontier`` to understand the task
  2. ``design`` — ~60% of the budget — generate code and iterate on
     ``validate_code`` results
  3. ``reserve`` — last 30s — guaranteed window to wrap up and submit a
     fallback template if nothing else worked

Stdout is reserved for the final JSON proposal the harness parses; all
diagnostic output goes to stderr.
"""
from __future__ import annotations

import json
import os
import sys
import time

from core.fallback_templates import fallback_name_for, generate_fallback
from core.history import extract_flops_budget, identify_bucket
from core.validation import validate_code

# Package mode (tests import ``agents.openai_sdk``) uses relative imports so
# the chat/tool modules resolve to ``agents.openai_sdk.*`` and test mocks
# patching those dotted paths actually apply. Standalone mode (harness loads
# this file via ``spec.loader.exec_module``) has no ``__package__``, so the
# relative form raises ``ImportError`` and we fall back to the sibling-on-
# sys.path form — the agent's directory is already on ``sys.path`` in both
# modes (harness adds it; ``__init__.py`` adds it in package mode).
try:
    from .llm_client import chat
    from .prompts import build_system_prompt, build_user_prompt
    from .tools import TOOLS, SubmitSignal, build_handlers
except ImportError:
    from llm_client import chat
    from prompts import build_system_prompt, build_user_prompt
    from tools import TOOLS, SubmitSignal, build_handlers


FALLBACK_RESERVE_SECONDS = 30
RESEARCH_BUDGET_FRAC = 0.20
RESEARCH_BUDGET_MAX = 120
RESEARCH_BUDGET_MIN = 60

# Small per-phase tool-loop caps so the LLM doesn't sprawl.
RESEARCH_MAX_ROUNDS = 4
DESIGN_MAX_ROUNDS = 8

DEFAULT_MODEL = "moonshotai/Kimi-K2.5-TEE"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _agent_budget(challenge: dict) -> int:
    """Resolve the seconds available to this agent.

    Preference order: ``challenge["agent_seconds"]`` (correct source) →
    ``AGENT_BUDGET_SECONDS`` env override → ``task.time_budget`` (the
    trainer's budget — wrong but better than nothing, with a loud warning).
    """
    b = int(challenge.get("agent_seconds") or 0)
    if b <= 0:
        try:
            b = int(os.environ.get("AGENT_BUDGET_SECONDS", "0") or 0)
        except ValueError:
            b = 0
    if b <= 0:
        task = challenge.get("task", {}) or {}
        b = int(task.get("time_budget", 300) or 300)
        _log(
            f"[agent] WARN: falling back to task.time_budget={b}s — that's "
            "the trainer's budget, not the agent's. Set "
            "challenge.agent_seconds or AGENT_BUDGET_SECONDS env."
        )
    return b


def _extract_code_block(text: str) -> str:
    """Pull the first fenced ```python block out of a text response.

    Safety net for when the LLM returns code in plain text instead of
    calling ``submit``.
    """
    if not text:
        return ""
    for marker in ("```python", "```Python", "```py"):
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.find("```", start)
            return (text[start:] if end == -1 else text[start:end]).strip()
    if "```" in text:
        start = text.index("```") + 3
        nl = text.find("\n", start)
        if nl == -1:
            return text[start:].strip()
        start = nl + 1
        end = text.find("```", start)
        return (text[start:] if end == -1 else text[start:end]).strip()
    return ""


def _package(code: str, name: str, motivation: str) -> dict:
    return {"code": code, "name": name, "motivation": motivation}


def _serialize_assistant_message(msg) -> dict:
    """Convert an SDK ChatCompletionMessage into a plain dict suitable for
    feeding back into the next chat call.

    The SDK's ``model_dump`` is the natural path; if it's not available
    (mocked client in tests), fall back to a hand-rolled dict.
    """
    if hasattr(msg, "model_dump"):
        try:
            return msg.model_dump(exclude_none=True)
        except Exception:
            pass
    out: dict = {"role": "assistant", "content": getattr(msg, "content", "") or ""}
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        serialised = []
        for tc in tool_calls:
            serialised.append({
                "id": getattr(tc, "id", ""),
                "type": getattr(tc, "type", "function"),
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            })
        out["tool_calls"] = serialised
    return out


def _run_tool_loop(
    *,
    messages: list[dict],
    tools: list[dict],
    handlers: dict,
    deadline: float,
    max_rounds: int,
    phase: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
):
    """Run tool-calling rounds until ``max_rounds``, deadline, or submit.

    Returns ``(updated_messages, submitted_code_or_None)``. ``submit`` is
    a terminal tool — if the model calls it, we stop and return the
    code argument.
    """
    submitted: str | None = None

    for round_num in range(max_rounds):
        remaining = deadline - time.monotonic()
        if remaining < 15:
            _log(
                f"[agent] {phase} round {round_num + 1}: "
                f"only {remaining:.0f}s left, stopping"
            )
            break

        try:
            resp = chat(
                messages=messages,
                tools=tools,
                model=model,
                temperature=temperature,
                max_tokens=4096,
                deadline=deadline,
            )
        except Exception as exc:
            _log(
                f"[agent] {phase} round {round_num + 1} chat failed: {exc}"
            )
            break

        try:
            msg = resp.choices[0].message
        except (AttributeError, IndexError) as exc:
            _log(f"[agent] {phase} malformed response: {exc}")
            break

        messages.append(_serialize_assistant_message(msg))

        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            content = getattr(msg, "content", "") or ""
            code = _extract_code_block(content)
            if code and phase == "design":
                _log(
                    f"[agent] {phase}: extracted code block from text "
                    f"({len(code)} chars)"
                )
                submitted = code
            break

        for tc in tool_calls:
            try:
                name = tc.function.name
                call_id = getattr(tc, "id", "") or ""
                raw_args = tc.function.arguments or "{}"
            except AttributeError as exc:
                _log(f"[agent] {phase} malformed tool call: {exc}")
                continue

            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}

            handler = handlers.get(name)
            if handler is None:
                result = f"unknown tool: {name}"
            else:
                try:
                    result = handler(**args)
                except SubmitSignal as sig:
                    submitted = sig.code
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": "submitted",
                    })
                    _log(
                        f"[agent] {phase}: model submitted "
                        f"({len(submitted)} chars, name={sig.name})"
                    )
                    return messages, submitted, sig
                except Exception as exc:
                    result = f"tool error: {exc}"

            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                # Cap to avoid context blow-up from giant tool results.
                "content": str(result)[:4000],
            })

    return messages, submitted, None


def design_architecture(challenge: dict, _gated_client) -> dict:
    """Entry point required by the harness.

    ``_gated_client`` is intentionally ignored — egress to the LLM proxy
    is enforced by iptables at the pod level, and using the OpenAI SDK
    directly keeps the agent code small. The other agents in this repo
    still use ``GatedClient``; this one doesn't.
    """
    t_start = time.monotonic()
    budget = _agent_budget(challenge)
    deadline = t_start + budget - FALLBACK_RESERVE_SECONDS
    research_window = max(
        RESEARCH_BUDGET_MIN,
        min(RESEARCH_BUDGET_MAX, int(budget * RESEARCH_BUDGET_FRAC)),
    )
    research_deadline = min(t_start + research_window, deadline)

    _log(
        f"[agent] start budget={budget}s "
        f"deadline_in={budget - FALLBACK_RESERVE_SECONDS}s "
        f"research_in={int(research_deadline - t_start)}s"
    )

    flops_min, flops_max = extract_flops_budget(challenge)
    bucket = identify_bucket(flops_min, flops_max)
    handlers = build_handlers(challenge)

    messages: list[dict] = [
        {"role": "system", "content": build_system_prompt(challenge, bucket)},
        {"role": "user", "content": build_user_prompt(challenge, bucket)},
    ]

    last_validated_code: str | None = None
    last_proposed_code: str | None = None
    submit_sig: SubmitSignal | None = None

    # ── Phase 1: research ────────────────────────────────────────────
    _log("[agent] phase=research")
    messages, _candidate, submit_sig = _run_tool_loop(
        messages=messages,
        tools=TOOLS,
        handlers=handlers,
        deadline=research_deadline,
        max_rounds=RESEARCH_MAX_ROUNDS,
        phase="research",
    )

    # If the model already submitted in research (eager), accept it and
    # skip straight to packaging.
    if submit_sig is None:
        # ── Phase 2: design ─────────────────────────────────────────
        _log("[agent] phase=design")
        messages.append({
            "role": "user",
            "content": (
                "Now propose a model. Call validate_code on your "
                "candidate; if it fails, iterate. When the code passes, "
                "call submit with the final code, a short name, and a "
                "motivation."
            ),
        })
        messages, candidate_code, submit_sig = _run_tool_loop(
            messages=messages,
            tools=TOOLS,
            handlers=handlers,
            deadline=deadline,
            max_rounds=DESIGN_MAX_ROUNDS,
            phase="design",
        )
        if candidate_code:
            ok, errors = validate_code(candidate_code, challenge)
            if ok:
                last_validated_code = candidate_code
            else:
                last_proposed_code = candidate_code
                _log(
                    f"[agent] candidate code didn't fully validate: {errors}"
                )

    # ── Phase 3: reserve / finalize ──────────────────────────────────
    elapsed = time.monotonic() - t_start
    _log(f"[agent] phase=reserve elapsed={elapsed:.0f}s")

    if submit_sig is not None:
        return _package(
            submit_sig.code, submit_sig.name, submit_sig.motivation,
        )
    if last_validated_code:
        return _package(
            last_validated_code,
            "openai_sdk_llm",
            "LLM-generated and validated",
        )
    if last_proposed_code:
        return _package(
            last_proposed_code,
            "openai_sdk_best_effort",
            "LLM-generated, structurally valid (FLOPs may be off)",
        )

    _log("[agent] no LLM code produced — using fallback template")
    try:
        fb_code = generate_fallback(challenge)
        fb_name = fallback_name_for(challenge)
        return _package(
            fb_code,
            fb_name,
            "OpenAI-SDK agent could not produce code; template fallback",
        )
    except Exception as exc:
        _log(f"[agent] fallback generation failed: {exc}")
        return _package(
            "",
            f"skipped_{bucket}",
            f"OpenAI-SDK agent failed; fallback also failed: {exc}",
        )
