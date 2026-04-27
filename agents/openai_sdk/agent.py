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
import tempfile
import time

from core import history
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
    from .llm_client import chat, get_client
    from .prompts import (
        build_system_prompt, build_turn_header, build_user_prompt,
    )
    from .tools import SubmitSignal, build_handlers, build_tools
except ImportError:
    from llm_client import chat, get_client
    from prompts import (
        build_system_prompt, build_turn_header, build_user_prompt,
    )
    from tools import SubmitSignal, build_handlers, build_tools


FALLBACK_RESERVE_SECONDS = 30

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


def _is_config_error(exc: BaseException) -> bool:
    """Detect config-class errors that won't resolve by retrying.

    ``get_client()`` raises ``RuntimeError`` when no URL / token is
    resolvable. ``KeyError('LLM_URL')`` can leak from legacy call paths.
    HTTP 401/403 from the proxy mean the agent token is missing or
    invalid — also not something retrying another phase will fix. All of
    these mean the agent cannot reach the LLM at all, so retrying across
    phases just wastes the round.
    """
    if isinstance(exc, KeyError) and str(exc) in ("'LLM_URL'", "'AGENT_TOKEN'"):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc)
        if "LLM URL" in msg or "agent token" in msg:
            return True
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return True
    s = str(exc).lower()
    if "invalid agent token" in s or "permission denied" in s:
        return True
    if type(exc).__name__ in ("PermissionDeniedError", "AuthenticationError"):
        return True
    return False


def _run_tool_loop(
    *,
    messages: list[dict],
    tools: list[dict],
    handlers: dict,
    deadline: float,
    phase: str,
    t_start: float,
    llm_url: str = "",
    agent_token: str = "",
    miner_uid: str = "",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
):
    """Run tool-calling rounds until the deadline approaches or the
    LLM submits.

    Returns ``(updated_messages, submitted_code, submit_sig, failure)``.
    ``failure`` is a short reason string when the loop couldn't produce
    anything useful (``"config: ..."``, ``"chat: timeout"``, etc.), or
    ``None`` on clean exit. The caller uses it to set an honest fallback
    motivation and, for config errors, to skip later work.
    """
    submitted: str | None = None
    failure: str | None = None
    rounds_since_validated = 0

    round_num = 0
    while True:
        round_num += 1
        remaining = deadline - time.monotonic()
        if remaining < 60:
            _log(
                f"[agent] {phase} round {round_num}: "
                f"only {remaining:.0f}s left, stopping"
            )
            if failure is None:
                failure = "deadline"
            break

        # Per-turn informational header — elapsed minutes + whether the
        # model has validated code on file. No directives; the LLM owns
        # its own pacing.
        has_validated = bool(getattr(
            handlers.get("submit", None), "_has_validated", False,
        ))
        elapsed_s = int(time.monotonic() - t_start)

        # Replace the previous turn-header instead of appending a stack
        # of stale status lines. Only safe when the immediate
        # predecessor is itself a turn-header user message AND the
        # message before it is not a tool message — the tool-call
        # protocol requires tool messages to immediately follow the
        # assistant's tool_calls, so we never pop a header that was
        # inserted between assistant tool_calls and tool results.
        if (
            len(messages) >= 2
            and messages[-1].get("role") == "user"
            and isinstance(messages[-1].get("content"), str)
            and messages[-1]["content"].startswith("[elapsed:")
            and messages[-2].get("role") != "tool"
        ):
            messages.pop()

        messages.append({
            "role": "user",
            "content": build_turn_header(
                elapsed_s=elapsed_s,
                has_validated=has_validated,
            ),
        })

        try:
            resp = chat(
                messages=messages,
                tools=tools,
                llm_url=llm_url,
                agent_token=agent_token,
                miner_uid=miner_uid,
                model=model,
                temperature=temperature,
                max_tokens=4096,
                deadline=deadline,
            )
        except Exception as exc:
            if _is_config_error(exc):
                _log(
                    f"[agent] {phase} round {round_num} "
                    f"CONFIG ERROR: {exc}"
                )
                failure = f"config: {exc}"
            else:
                _log(
                    f"[agent] {phase} round {round_num} "
                    f"chat failed: {exc}"
                )
                failure = f"chat: {type(exc).__name__}: {exc}"
            break

        try:
            msg = resp.choices[0].message
        except (AttributeError, IndexError) as exc:
            _log(f"[agent] {phase} malformed response: {exc}")
            failure = f"chat: malformed response ({exc})"
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
                    return messages, submitted, sig, None
                except Exception as exc:
                    result = f"tool error: {exc}"

            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                # Cap to avoid context blow-up from giant tool results.
                "content": str(result)[:4000],
            })

        # If validate_code just succeeded this round, hit the LLM with
        # a dedicated user-role nudge right at the boundary — the turn
        # header alone has not been enough to break the
        # validate-then-keep-sketching loop. Only fire this when the
        # validate call moved us from "no validated code" to "have
        # validated code" so we don't spam every subsequent round.
        validated_this_round = (
            any(
                getattr(getattr(tc, "function", None), "name", None)
                == "validate_code"
                for tc in tool_calls
            )
            and bool(getattr(
                handlers.get("submit", None), "_has_validated", False,
            ))
        )
        if validated_this_round and phase == "design":
            messages.append({
                "role": "user",
                "content": (
                    "Validation passed. Submit now. Your next tool "
                    "call must be `write_scratchpad` then `submit` "
                    "with the validated code. Do not produce another "
                    "candidate."
                ),
            })

        # Validated-and-stalled early exit. After validation the model
        # only needs one more turn to ship; if it burns two more
        # rounds without submitting we bail and let the agent's
        # auto-submit recovery path package the validated code.
        has_validated_now = bool(getattr(
            handlers.get("submit", None), "_has_validated", False,
        ))
        if has_validated_now:
            rounds_since_validated += 1
            if rounds_since_validated >= 2:
                _log(
                    f"[agent] {phase}: 2 rounds since validation, "
                    "no submit — breaking to auto-submit"
                )
                break

    return messages, submitted, None, failure


def design_architecture(challenge: dict, gated_client=None) -> dict:
    """Entry point required by the harness.

    LLM-proxy egress is enforced by iptables at the pod level, so the
    agent talks to the LLM through the OpenAI SDK directly. ``gated_client``
    IS used for the research tools (``search_papers``, ``query_db``) and
    is the only allowed HTTP transport to the experiment DB and arxiv
    mirror — those endpoints are not behind the iptables allowlist for
    direct SDK traffic.
    """
    t_start = time.monotonic()
    budget = _agent_budget(challenge)
    deadline = t_start + budget - FALLBACK_RESERVE_SECONDS

    _log(
        f"[agent] start budget={budget}s "
        f"deadline_in={budget - FALLBACK_RESERVE_SECONDS}s"
    )

    flops_min, flops_max = extract_flops_budget(challenge)
    bucket = identify_bucket(flops_min, flops_max)

    # ── Scratchpad load (via harness-injected globals) ────────────────
    # The harness injects ``load_scratchpad``/``save_scratchpad`` into
    # the agent module's globals. Outside the harness (unit tests,
    # manual runs) those names don't exist, so catch ``NameError`` and
    # run without scratchpad state.
    scratch_dir: str | None = None
    try:
        scratch_dir = load_scratchpad(challenge)  # noqa: F821 — injected
    except NameError:
        _log("[agent] load_scratchpad not injected — running without scratchpad")
    except Exception as exc:
        _log(f"[agent] scratchpad load failed: {exc}")

    # Load state up front so ``previous_results`` (round-score feedback
    # delivered by the harness on the next round) can be merged before
    # prompts are built or handlers see history.
    state = history.load_state(scratch_dir) if scratch_dir else {}
    prev_results = challenge.get("previous_results") or []
    score_direction = challenge.get("score_direction") or "minimize"
    if prev_results:
        history.merge_results_into_state(state, prev_results)
    all_hist = history.get_history(state)
    scored = [
        e for e in all_hist if isinstance(e.get("score"), (int, float))
    ]
    pending = len(all_hist) - len(scored)
    best = history.best_own_submission(state, score_direction)
    if best is not None:
        rank = best.get("rank", "?")
        total = best.get("rank_total", "?")
        _log(
            f"[agent] score feedback: {len(prev_results)} prior merged, "
            f"best own score={best['score']:.4g} (rank {rank}/{total}), "
            f"{pending} pending"
        )
    else:
        _log(
            f"[agent] score feedback: {len(prev_results)} prior merged, "
            f"no scored submissions yet, {pending} pending"
        )

    handlers = build_handlers(
        challenge,
        client=gated_client,
        scratch_dir=scratch_dir,
        deadline=deadline,
        state=state,
    )
    tools = build_tools(challenge)
    llm_url = challenge.get("llm_url", "") or ""
    # The harness injects the token into challenge["agent_token"], not
    # into the pod env — read it from there so every get_client/chat
    # call has the credential and the proxy doesn't 403.
    agent_token = challenge.get("agent_token", "") or ""
    miner_uid = str(challenge.get("miner_uid", "") or "")

    # ── Startup config check ─────────────────────────────────────────
    # Build the client once up front so a config failure (missing URL,
    # bad token) shows as a distinct diagnostic instead of masquerading
    # as a per-phase chat error on every phase.
    config_broken = False
    config_error: str | None = None
    try:
        get_client(llm_url, agent_token, miner_uid)
    except RuntimeError as exc:
        _log(f"[agent] startup config check failed: {exc}")
        config_broken = True
        config_error = f"config error: {exc}"

    messages: list[dict] = [
        {"role": "system", "content": build_system_prompt(challenge, bucket)},
        {"role": "user", "content": build_user_prompt(challenge, bucket)},
    ]

    last_validated_code: str | None = None
    last_proposed_code: str | None = None
    last_validation_errors: list[str] = []
    submit_sig: SubmitSignal | None = None
    design_failure: str | None = None

    if config_broken:
        # Nothing we can do without an LLM — skip straight to packaging.
        _log("[agent] config broken, skipping main loop")
    else:
        messages, candidate_code, submit_sig, design_failure = (
            _run_tool_loop(
                messages=messages,
                tools=tools,
                handlers=handlers,
                deadline=deadline,
                phase="main",
                t_start=t_start,
                llm_url=llm_url,
                agent_token=agent_token,
                miner_uid=miner_uid,
            )
        )
        if design_failure and design_failure.startswith("config:"):
            config_broken = True
            config_error = design_failure.replace(
                "config:", "config error:", 1,
            )
        if candidate_code:
            ok, errors = validate_code(candidate_code, challenge)
            if ok:
                last_validated_code = candidate_code
            else:
                last_proposed_code = candidate_code
                last_validation_errors = list(errors)
                _log(
                    f"[agent] candidate code didn't fully validate: "
                    f"{errors}"
                )

    # ── Scratchpad save ──────────────────────────────────────────────
    # Persist any notes the LLM stashed via ``write_scratchpad`` before
    # we spend the reserve window on packaging. The submit handler is
    # exempt from circuit-breaker wrapping but still exposes the
    # ``_state_holder`` attribute via the wrapped callable.
    try:
        state_holder = getattr(
            handlers.get("submit", None), "_state_holder", None,
        )
        if state_holder is not None:
            scratch_dir = scratch_dir or tempfile.mkdtemp()
            history.save_state(scratch_dir, state_holder["state"])
            try:
                save_scratchpad(challenge, scratch_dir)  # noqa: F821 — injected
            except NameError:
                pass  # Outside the harness — nothing to upload.
            except Exception as exc:
                _log(f"[agent] scratchpad save failed: {exc}")
    except Exception as exc:
        _log(f"[agent] scratchpad finalize crashed: {exc}")

    # ── Phase 3: reserve / finalize ──────────────────────────────────
    elapsed = time.monotonic() - t_start
    _log(f"[agent] phase=reserve elapsed={elapsed:.0f}s")

    # Per-round tool-usage summary. Stashed on the submit wrapper so
    # it's a stable single attach-point (alongside ``_state_holder``).
    call_counts = getattr(
        handlers.get("submit", None), "_call_counts", None,
    )
    if call_counts:
        total = sum(call_counts.values())
        summary = ", ".join(
            f"{name}={count}"
            for name, count in sorted(
                call_counts.items(), key=lambda kv: (-kv[1], kv[0]),
            )
        )
        _log(f"[agent] tool calls (total={total}): {summary}")
    else:
        _log("[agent] tool calls: none")

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
        joined = "; ".join(last_validation_errors)[:200]
        return _package(
            last_proposed_code,
            "openai_sdk_best_effort",
            f"LLM code failed validation: {joined}",
        )

    # Pick the most specific failure motivation we have.
    if config_error:
        fallback_motivation = config_error
    else:
        failure = design_failure
        if failure and failure.startswith("chat:"):
            detail = failure.split(":", 1)[1].strip()
            if "timeout" in detail.lower() or "timed out" in detail.lower():
                fallback_motivation = "LLM timeout"
            else:
                fallback_motivation = f"LLM chat failed: {detail}"
        elif failure == "deadline":
            fallback_motivation = (
                "LLM returned no parseable code before deadline"
            )
        else:
            fallback_motivation = (
                f"LLM returned no parseable code after "
                f"{RESEARCH_MAX_ROUNDS + DESIGN_MAX_ROUNDS} rounds"
            )

    # Stage A: auto-submit recovery. If the LLM called validate_code
    # successfully but never followed up with submit, ship that
    # validated code instead of going home empty-handed.
    auto_code = getattr(
        handlers.get("submit", None), "_last_validated_code", "",
    ) or ""
    if auto_code:
        _log(
            f"[agent] auto-submit: LLM never called submit but validated "
            f"code exists ({len(auto_code)} chars)"
        )
        return _package(
            auto_code,
            f"auto_submit_{bucket}",
            "Auto-submitted validated code — LLM did not call submit "
            f"explicitly. Root cause: {fallback_motivation}",
        )

    # Stage B: honest failure. No template fallback — return empty code
    # so the failure is visible in the round results instead of being
    # masked by a near-zero MLP.
    _log(
        f"[agent] FAILED — no code produced. "
        f"Root cause: {fallback_motivation}"
    )
    return _package(
        "",
        f"failed_{bucket}",
        f"FAILURE: {fallback_motivation}",
    )
