"""Autonomous Agent — tool-using LLM that loops until it has a validated model.

Unlike the rigid pipeline agents, this agent:
  1. Receives a system prompt describing the goal, budget, and available APIs
  2. Gets a rich set of tools (research, validate, estimate, submit, etc.)
  3. Decides autonomously what to do each turn — research, generate, iterate
  4. Loops until it calls ``submit`` with validated code, or time runs out

The LLM is the controller. It chooses when to research papers, query the DB,
look at the frontier, generate code, validate it, fix errors, and submit.
"""

import json
import sys
import tempfile
import time

from core import history, validation
from core.arch_knowledge import build_arch_guidance
from core.fallback_templates import generate_fallback
from core.history import extract_flops_budget, identify_bucket
from core.prompt_builder import _format_task_params, _compute_sizing_guidance
from strategies import build_strategy, select_strategy
from tools import TOOLS, SubmitSignal, build_handlers, build_tools

DEFAULT_MODEL = "moonshotai/Kimi-K2.5-TEE"

# Reserve ~5 LLM calls worth of headroom from the 30-request rate limit.
# Each loop iteration is 1 call, so this gives us up to 25 turns.
MAX_TURNS = 25

# Minimum seconds to reserve for final submission overhead.
TIME_BUFFER_SECONDS = 10


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _extract_code_block(text: str) -> str:
    """Extract the first fenced Python code block from an LLM text response.

    Used as a safety net when the LLM answers with code in plain text instead
    of calling ``validate_code``/``submit``. Returns "" if no recognizable
    block is found.
    """
    if not text:
        return ""
    for marker in ("```python", "```Python", "```py"):
        if marker in text:
            start = text.index(marker) + len(marker)
            closing = text.find("```", start)
            return (text[start:] if closing == -1 else text[start:closing]).strip()
    if "```" in text:
        start = text.index("```") + 3
        nl = text.find("\n", start)
        if nl == -1:
            return text[start:].strip()
        start = nl + 1
        closing = text.find("```", start)
        return (text[start:] if closing == -1 else text[start:closing]).strip()
    return ""


def _build_system_prompt(challenge: dict, strategy: dict | None = None) -> str:
    """Build a system prompt that teaches the LLM how to be an autonomous agent.

    ``strategy`` (optional) is a strategy dict from ``strategies.build_strategy``
    — it contributes an identity persona and workflow hints without changing
    any of the tools, validation, or loop structure.
    """
    strategy = strategy or {}
    task = challenge.get("task", {})
    tp = task.get("task_params", {})
    param_str = ", ".join(tp.keys()) if tp else "**task_params"
    task_name = task.get("name", "unknown")
    flops_min, flops_max = extract_flops_budget(challenge)
    bucket = identify_bucket(flops_min, flops_max)
    target = int(flops_max * 0.6) if flops_max else 0
    gate_min = int(flops_min * 0.9) if flops_min else 0
    gate_max = int(flops_max * 1.1) if flops_max else 0

    parts = []

    # ── Identity & goal ───────────────────────────────────────────
    parts.append(
        "You are an autonomous ML architecture designer. Your goal is to "
        "produce the best possible model code for a competitive evaluation.\n\n"
        "You have tools to research, analyze, generate, validate, and submit. "
        "You decide the strategy. Use the tools in whatever order makes sense. "
        "The loop ends when you call `submit` with validated code, or time runs out."
    )

    # ── Strategy identity (persona) ───────────────────────────────
    if strategy.get("identity"):
        parts.append(f"## Strategy Persona\n{strategy['identity']}")

    # ── Task description ──────────────────────────────────────────
    desc = task.get("description", "")
    parts.append(
        f"## Task: {task_name}\n"
        + (f"{desc}\n\n" if desc else "")
        + f"Task parameters: {_format_task_params(tp)}\n"
        f"build_model signature: `def build_model({param_str})`"
    )

    # ── Architecture reasoning guidance ───────────────────────────
    frontier = (
        challenge.get("feasible_frontier")
        or challenge.get("pareto_frontier")
        or []
    )
    parts.append(build_arch_guidance(challenge, frontier))

    # ── Domain context ────────────────────────────────────────────
    domain = task.get("domain_system_prompt", "")
    if domain:
        parts.append(f"## Domain Context\n{domain}")

    # ── Constraints & objectives ──────────────────────────────────
    constraints = task.get("constraints", [])
    if constraints:
        parts.append("## Constraints\n" + "\n".join(f"- {c}" for c in constraints))

    objectives = task.get("objectives", [])
    if objectives:
        parts.append("## Objectives\n" + "\n".join(f"- {o}" for o in objectives))

    anti_patterns = task.get("anti_patterns", [])
    if anti_patterns:
        parts.append("## Anti-Patterns\n" + "\n".join(f"- {a}" for a in anti_patterns))

    example_hypotheses = task.get("example_hypotheses", [])
    if example_hypotheses:
        parts.append(
            "## Example Hypotheses (inspiration, not prescriptions)\n"
            + "\n".join(f"- {h}" for h in example_hypotheses)
        )

    # ── Budget ────────────────────────────────────────────────────
    parts.append(
        f"## FLOPs Budget\n"
        f"Bucket: {bucket}\n"
        f"Range: [{flops_min:,}, {flops_max:,}]\n"
        f"Target: ~{target:,} FLOPs (60% of max)\n"
        f"Hard gate: [{gate_min:,}, {gate_max:,}] (instant rejection outside this)"
    )

    # ── Sizing guidance ───────────────────────────────────────────
    parts.append(_compute_sizing_guidance(challenge))

    # ── Code requirements ─────────────────────────────────────────
    parts.append(
        "## Code Requirements\n"
        f"1. `def build_model({param_str})` — top-level, returns nn.Module\n"
        "2. `def build_optimizer(model)` — top-level, returns Optimizer\n"
        "3. Optional hooks: training_config(), init_weights(), configure_amp(), "
        "build_scheduler(), compute_loss(), transform_batch(), on_step_end()\n"
        "4. Only torch + stdlib — no external dependencies\n"
        "5. No subprocess, socket, or ftplib imports\n"
        "6. Use standard nn ops (Linear, Conv1d, MultiheadAttention, etc.) "
        "so the FLOPs counter works\n"
        "7. Read ALL task parameters from the function arguments — NEVER "
        "hardcode dimension values, output shapes, or task-specific assumptions"
    )

    # ── CRITICAL: output shape ────────────────────────────────────
    parts.append(
        "## CRITICAL: Output Shape\n"
        "The #1 cause of training failure is tensor size mismatch — the "
        "model projects to the wrong dim and the loss function errors out "
        "with 'size of tensor a (X) must match the size of tensor b (Y)' "
        "partway through training.\n\n"
        "Your model's `forward()` MUST return a tensor whose shape EXACTLY "
        "matches the shape described in the task constraints. Every "
        "non-batch dimension must be derived from the build_model "
        "arguments (task_params) — NEVER hardcode a dimension and NEVER "
        "confuse input-side dimensions with output-side dimensions. Read "
        "the constraint string carefully to determine which task_param "
        "maps to which output dimension. The `check_output_shape` tool "
        "will catch mismatches.\n\n"
        "ALWAYS call `validate_code` before `submit`. `validate_code` runs "
        "a real forward pass and checks the output shape against the "
        "constraint — if it reports a dimension mismatch you MUST fix it "
        "and re-validate. If `validate_code` reports a shape error, the "
        "`check_output_shape` tool can help diagnose which layer produces "
        "the wrong dim. Submitting code that hasn't passed `validate_code` "
        "is the fastest way to lose this round."
    )

    # ── Workflow guidance ─────────────────────────────────────────
    parts.append(
        "## Recommended Workflow\n"
        "1. Call `analyze_task` to get shapes, budget, and frontier gaps in "
        "one structured summary\n"
        "2. Call `read_scratchpad` for notes from previous rounds\n"
        "3. Optionally research: `get_frontier_details`, `query_db`, "
        "`search_papers` (don't spend more than 2-3 turns on research)\n"
        "4. DESIGN ITERATIVELY:\n"
        "   a. Use `estimate_layer_flops` to cost individual building blocks "
        "you're considering. Any PyTorch module works — standard or custom.\n"
        "   b. Use `sketch_architecture` to test your overall design. Write "
        "a minimal build_model, get back per-layer FLOPs and shape flow. "
        "Iterate until the shapes are right and FLOPs fit the budget.\n"
        "   This is MUCH cheaper than writing full code and debugging failures.\n"
        "5. Expand your validated sketch into full code: add build_optimizer, "
        "training_config, build_scheduler, init_weights, and other hooks.\n"
        "6. Call `validate_code` — final check on syntax, FLOPs, AND output shape.\n"
        "7. If validation fails, fix based on the error and re-validate.\n"
        "8. Call `submit` with validated code.\n"
        "9. Save notes via `write_scratchpad` for future rounds.\n\n"
        "This is a suggestion — you can do these in any order. "
        "You can also skip the prototype step and go straight to full code "
        "if you're confident. But prototyping with `estimate_layer_flops` and "
        "`sketch_architecture` is cheap and catches mistakes early.\n\n"
        "## TIME MANAGEMENT (CRITICAL)\n"
        "- You should have validated code by 85% of your time budget. "
        "Use the earlier portion for research, prototyping, and iterating on your design.\n"
        "- Generate and validate a working model FIRST, then iterate to improve\n"
        "- If validate_code fails on FLOPs, follow the resize suggestion immediately\n"
        "- If time is running low, submit what you have rather than researching more\n"
        "- A submitted model that passes validation always beats no submission"
    )

    # ── Strategy-specific workflow (appended, not replacing) ──────
    if strategy.get("workflow_guidance"):
        parts.append(strategy["workflow_guidance"])

    return "\n\n".join(parts)


def _build_kickoff_message(challenge: dict, strategy: dict | None = None) -> str:
    """Build the initial user message that starts the agent loop.

    ``strategy['kickoff_additions']`` is appended verbatim to focus the
    agent on the strategy-specific angle for this round.
    """
    strategy = strategy or {}
    task = challenge.get("task", {})
    task_name = task.get("name", "unknown")
    flops_min, flops_max = extract_flops_budget(challenge)
    bucket = identify_bucket(flops_min, flops_max)
    frontier = challenge.get("feasible_frontier", [])
    if not frontier:
        frontier = challenge.get("pareto_frontier", [])

    time_budget = task.get("time_budget", 300)

    base = (
        f"New round started. Task: {task_name}, Bucket: {bucket}, "
        f"FLOPs range: [{flops_min:,}, {flops_max:,}].\n"
        f"Frontier has {len(frontier)} model(s) to beat.\n"
        f"Time budget: {time_budget}s.\n\n"
        "Begin your autonomous design process. Use the tools available to "
        "research, design, validate, and submit the best model you can."
    )
    additions = strategy.get("kickoff_additions", "")
    if additions:
        return f"{base}\n\n{additions}"
    return base


def _structural_ok(code: str, challenge: dict | None) -> tuple[bool, list[str]]:
    """Lenient structural check used for final fallback acceptance.

    Delegates to ``validation.validate_code`` but without the FLOPs hard-gate
    path: we pass ``challenge=None`` so the estimator is skipped and only
    parseability + top-level ``build_model``/``build_optimizer`` + forbidden-
    import checks are applied. If the code clears those bars the harness will
    accept it at ``pre_validate_code`` time; any FLOPs mismatch is handled by
    the harness's own scoring penalty rather than an instant rejection. This
    means the agent never has to submit literally empty code just because its
    proposed model landed slightly outside the FLOPs budget.
    """
    if not code or not code.strip():
        return False, ["Empty code"]
    return validation.validate_code(code, challenge=None)


def _autonomous_loop(client, challenge: dict, messages: list[dict],
                     tool_handlers: dict, deadline: float,
                     tools: list | None = None,
                     temperature: float = 0.7) -> dict | None:
    """Run the autonomous tool-calling loop.

    Returns the submission dict {"code", "name", "motivation"} if the agent
    calls submit, or None if time/turns run out without a submission.

    Time management checkpoints:
      - At 85% time used with no validated code: inject urgent nudge
      - At 93% time used with no validated code: auto-generate fallback
    """
    llm_url = challenge.get("llm_url", "")
    if not llm_url:
        _log("[agent] No llm_url — cannot run autonomous loop")
        return None

    url = f"{llm_url}/v1/chat/completions"
    max_retries = 3
    last_validated_code = None     # Passed validate_code cleanly (preferred)
    last_proposed_code = None      # Structurally ok but may have failed FLOPs

    start_time = time.time()
    total_budget = deadline - start_time
    nudged_85pct = False           # Have we injected the 85% warning?
    fallback_injected = False      # Have we injected the 93% fallback?
    validation_failures: list[str] = []  # Track error strings across turns

    for turn in range(MAX_TURNS):
        remaining = deadline - time.time()
        if remaining < TIME_BUFFER_SECONDS:
            _log(f"[agent] Time's up at turn {turn + 1}, breaking loop")
            break

        elapsed = time.time() - start_time
        pct_used = elapsed / total_budget if total_budget > 0 else 1.0

        # ── 85% checkpoint: force code generation if nothing yet ──
        if pct_used >= 0.85 and not nudged_85pct and not last_validated_code:
            nudged_85pct = True
            _log("[agent] 85% time checkpoint — no validated code yet, injecting urgency")
            messages.append({
                "role": "user",
                "content": (
                    "WARNING: You have used 85% of your time with no validated "
                    "code. Generate and validate code NOW. You can iterate after "
                    "you have a working baseline. Call validate_code with your "
                    "best code immediately."
                ),
            })

        # ── 93% checkpoint: auto-generate fallback if still nothing ──
        if pct_used >= 0.93 and not fallback_injected and not last_validated_code:
            fallback_injected = True
            _log("[agent] 93% time checkpoint — generating fallback template")
            try:
                fb_code = generate_fallback(challenge)
                ok, errors = validation.validate_code(fb_code, challenge)
                if ok:
                    last_validated_code = fb_code
                    _log("[agent] Fallback template passed validation")
                    messages.append({
                        "role": "user",
                        "content": (
                            "URGENT: A fallback model has been auto-generated and "
                            "validated. You can still submit better code, but you "
                            "MUST call submit within the next 1-2 turns or the "
                            "fallback will be used. If you have improved code, "
                            "validate and submit it now."
                        ),
                    })
                else:
                    struct_ok, _ = _structural_ok(fb_code, challenge)
                    if struct_ok:
                        last_proposed_code = fb_code
                        _log(f"[agent] Fallback structurally ok but FLOPs off: {errors}")
                    else:
                        _log(f"[agent] Fallback template failed: {errors}")
            except Exception as exc:
                _log(f"[agent] Fallback generation failed: {exc}")

        # Build payload — include tools unless it's the very last turn
        payload = {
            "model": DEFAULT_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4096,
        }
        if turn < MAX_TURNS - 1:
            payload["tools"] = tools if tools is not None else TOOLS

        _log(f"[agent] Turn {turn + 1}/{MAX_TURNS}, "
             f"{remaining:.0f}s remaining, {len(messages)} messages")

        # LLM call with retries for transient failures
        resp = None
        for attempt in range(max_retries):
            try:
                resp = client.post_json(url, payload)
                break
            except Exception as exc:
                _log(f"[agent] Turn {turn + 1} attempt {attempt + 1}/{max_retries} "
                     f"failed: {exc}")

        if resp is None:
            _log(f"[agent] All retries failed at turn {turn + 1}")
            break

        choice = resp["choices"][0]
        assistant_msg = choice["message"]
        finish_reason = choice.get("finish_reason", "")
        tool_calls = assistant_msg.get("tool_calls")

        # No tool calls → the LLM wants to talk. Append and continue.
        # NOTE: we do NOT also branch on ``finish_reason == "stop"`` — some
        # OpenAI-compatible servers (notably certain Kimi deployments) return
        # ``finish_reason="stop"`` alongside a populated ``tool_calls`` list.
        # Short-circuiting on finish_reason would silently drop those tool
        # calls and leave an assistant message with unresolved tool_calls in
        # the history, which the next /v1/chat/completions call would reject.
        if not tool_calls:
            content = assistant_msg.get("content") or ""
            _log(f"[agent] Text response ({len(content)} chars, "
                 f"finish_reason={finish_reason}): {content[:200]}...")
            messages.append(assistant_msg)

            # Fallback: if the LLM dumped code in a text response instead of
            # calling validate_code, capture it so the round still has
            # something to submit. Track structurally-ok code even when the
            # FLOPs gate rejects it — the harness pre_validate only checks
            # structure, so shipping a mis-sized model beats shipping nothing.
            fallback_code = _extract_code_block(content)
            if fallback_code:
                ok, errors = validation.validate_code(fallback_code, challenge)
                if ok:
                    _log("[agent] Text response contained fully valid code")
                    last_validated_code = fallback_code
                    last_proposed_code = fallback_code
                else:
                    struct_ok, _ = _structural_ok(fallback_code, challenge)
                    if struct_ok:
                        _log("[agent] Text code block is structurally ok "
                             f"(full validation failed: {errors}) — "
                             "tracked as last_proposed_code")
                        last_proposed_code = fallback_code
                    else:
                        _log(f"[agent] Text code block unusable: {errors}")

            # If this is the last turn, the loop ends
            if turn == MAX_TURNS - 1:
                break
            # Nudge the agent to take action
            messages.append({
                "role": "user",
                "content": (
                    "Continue using your tools. If you have code ready, "
                    "call validate_code on it, then submit. "
                    "Check time_remaining if unsure how much time you have left."
                ),
            })
            continue

        # Process tool calls
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
                result_str = f"Unknown tool: '{fn_name}'. Available tools: {', '.join(tool_handlers.keys())}"
                _log(f"[agent] Unknown tool: {fn_name}")
            else:
                # Capture any code the LLM ran through validate_code/submit so
                # we have a fallback when the loop ends without a successful
                # submit — even if full validation failed on FLOPs grounds.
                if fn_name in ("validate_code", "submit"):
                    candidate = kwargs.get("code", "")
                    if candidate:
                        struct_ok, _ = _structural_ok(candidate, challenge)
                        if struct_ok:
                            last_proposed_code = candidate
                try:
                    result_str = str(handler(**kwargs))
                    _log(f"[agent] Tool {fn_name} → {len(result_str)} chars")

                    # Track fully-validated code separately (preferred path)
                    if fn_name == "validate_code" and result_str.startswith("PASSED"):
                        last_validated_code = kwargs.get("code", "")
                        validation_failures.clear()  # Reset on success

                    # Track repeated validation failures to escalate below
                    if fn_name == "validate_code" and result_str.startswith("FAILED"):
                        validation_failures.append(result_str)
                except SubmitSignal as sig:
                    _log(f"[agent] SUBMITTED: {sig.name}")
                    return {
                        "code": sig.code,
                        "name": sig.name,
                        "motivation": sig.motivation,
                    }
                except Exception as exc:
                    result_str = f"Tool error: {exc}"
                    _log(f"[agent] Tool {fn_name} raised: {exc}")

            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": result_str,
            })

            # ── Escalate on repeated identical validation failures ──
            if (
                fn_name == "validate_code"
                and result_str.startswith("FAILED")
                and len(validation_failures) >= 3
            ):
                recent = validation_failures[-3:]
                # Extract the first error line from each failure
                first_errors = []
                for f in recent:
                    lines = [l.strip("- ") for l in f.split("\n") if l.startswith("- ")]
                    if lines:
                        first_errors.append(lines[0])

                # If the same error appears in all 3 recent failures
                if (
                    len(first_errors) >= 3
                    and first_errors[-1] == first_errors[-2] == first_errors[-3]
                ):
                    _log(f"[agent] Same error 3x in a row: {first_errors[-1][:100]}")
                    messages.append({
                        "role": "user",
                        "content": (
                            "ESCALATION: You have hit the SAME validation error 3 times "
                            "in a row. Your current approach is not working. Do NOT make "
                            "another small tweak — step back and try a FUNDAMENTALLY "
                            "different approach:\n"
                            f"Repeated error: {first_errors[-1]}\n\n"
                            "Options:\n"
                            "- If it's a shape mismatch: use `trace_architecture` to see "
                            "where the shape goes wrong, or use `check_output_shape` for "
                            "a targeted diagnosis\n"
                            "- If it's a FLOPs error: use `estimate_layer_flops` to cost "
                            "individual layers and redesign from scratch\n"
                            "- If it's a structural error: re-read the task constraints "
                            "and start with a minimal skeleton\n"
                            "- Consider using `analyze_task` to re-examine the task "
                            "requirements if you haven't already"
                        ),
                    })

    # Loop ended without an explicit submit. Prefer fully-validated code;
    # otherwise try to fully validate any structurally-ok proposal before
    # falling back. Shipping code that only passed ``_structural_ok`` —
    # i.e. parses + has build_model/build_optimizer — is how we end up
    # submitting models with the wrong output shape or FLOPs mismatch,
    # which then crash mid-training. Better to fall through to the
    # guaranteed-valid template than to ship unvalidated code.
    if last_validated_code:
        _log("[agent] Loop ended without submit, using last validated code")
        return {
            "code": last_validated_code,
            "name": "autonomous_fallback",
            "motivation": "Time/turns exhausted — submitting last validated code",
        }
    if last_proposed_code:
        full_ok, full_errors = validation.validate_code(
            last_proposed_code, challenge,
        )
        if full_ok:
            _log("[agent] Last proposed code passed full validation on final check; "
                 "submitting it")
            return {
                "code": last_proposed_code,
                "name": "autonomous_best_effort",
                "motivation": (
                    "Time/turns exhausted; last proposed code passed full "
                    "validation on final re-check"
                ),
            }
        _log("[agent] Last proposed code failed full validation on final check "
             f"({full_errors}); falling through to guaranteed-valid template")

    _log("[agent] Loop ended with no fully-validated code")
    return None


def design_architecture(challenge: dict, client) -> dict:
    """Entry point called by the harness."""

    # ── Identify budget ───────────────────────────────────────────
    flops_min, flops_max = extract_flops_budget(challenge)
    bucket = identify_bucket(flops_min, flops_max)
    _log(f"[agent] Autonomous agent starting — bucket={bucket}, "
         f"FLOPs=[{flops_min:,}, {flops_max:,}]")

    # ── Time budget ───────────────────────────────────────────────
    task = challenge.get("task", {})
    time_budget = task.get("time_budget", 300)
    deadline = time.time() + time_budget
    _log(f"[agent] Time budget: {time_budget}s")

    # ── Load scratchpad ───────────────────────────────────────────
    scratch_dir = None
    try:
        scratch_dir = load_scratchpad(challenge)  # noqa: F821 — injected global
    except Exception as exc:
        _log(f"[agent] Scratchpad load failed: {exc}")

    state = history.load_state(scratch_dir) if scratch_dir else {}

    # ── Select strategy ───────────────────────────────────────────
    try:
        strategy_name = select_strategy(challenge, state)
        strategy = build_strategy(strategy_name, challenge, state)
    except Exception as exc:
        _log(f"[agent] Strategy selection failed: {exc} — using defaults")
        strategy_name = "simple_modeler"
        strategy = {
            "identity": "",
            "kickoff_additions": "",
            "workflow_guidance": "",
            "temperature": 0.7,
        }
    _log(f"[agent] Strategy: {strategy_name} (temp={strategy.get('temperature', 0.7)})")

    # ── Build tools ───────────────────────────────────────────────
    tool_handlers = build_handlers(client, challenge, scratch_dir, deadline)
    tools = build_tools(challenge)

    # ── Build messages ────────────────────────────────────────────
    system_prompt = _build_system_prompt(challenge, strategy)
    kickoff = _build_kickoff_message(challenge, strategy)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": kickoff},
    ]

    # ── Run autonomous loop ───────────────────────────────────────
    result = None
    llm_url = challenge.get("llm_url", "")
    if llm_url:
        try:
            result = _autonomous_loop(
                client, challenge, messages, tool_handlers, deadline,
                tools=tools,
                temperature=strategy.get("temperature", 0.7),
            )
        except Exception as exc:
            _log(f"[agent] Autonomous loop failed: {exc}")

    # ── Prepare output ────────────────────────────────────────────
    if result:
        code = result["code"]
        name = result["name"]
        motivation = result["motivation"]
        _log(f"[agent] Submitting: {name}")
    else:
        # Last-resort fallback: generate a guaranteed-valid model
        _log(f"[agent] No result from loop — trying fallback template for {bucket}")
        try:
            fb_code = generate_fallback(challenge)
            fb_ok, fb_errors = _structural_ok(fb_code, challenge)
            if fb_ok:
                code = fb_code
                name = f"fallback_{bucket}"
                motivation = (
                    "Autonomous agent could not produce LLM-generated code; "
                    "using auto-sized fallback template"
                )
                _log(f"[agent] Fallback template accepted for {bucket}")
            else:
                _log(f"[agent] Fallback template also failed: {fb_errors}")
                code = ""
                name = f"skipped_{bucket}"
                motivation = "Autonomous agent could not produce validated code"
        except Exception as exc:
            _log(f"[agent] Fallback generation failed: {exc}")
            code = ""
            name = f"skipped_{bucket}"
            motivation = "Autonomous agent could not produce validated code"

    # ── Final validation safety net ───────────────────────────────
    # We only wipe the submission when the code is STRUCTURALLY unusable
    # (won't parse, missing build_model/build_optimizer, forbidden import).
    # FLOPs mismatches are left for the harness to penalize rather than
    # instantly rejecting our own submission — shipping a mis-sized model
    # still passes pre_validate_code whereas empty code does not.
    if code:
        ok, errors = _structural_ok(code, challenge)
        if not ok:
            _log(f"[agent] Final structural validation failed: {errors}")
            code = ""
            name = f"skipped_{bucket}"
            motivation = f"Final validation failed: {errors}"
        else:
            # Log any non-structural issues (FLOPs, etc.) so we can diagnose
            # but still proceed with submission.
            full_ok, full_errors = validation.validate_code(code, challenge)
            if not full_ok:
                _log(f"[agent] Submitting despite non-structural issues: "
                     f"{full_errors}")

    # ── Save scratchpad ───────────────────────────────────────────
    # Prefer the tool-handler state (it may have been updated by the agent via
    # write_scratchpad); otherwise keep the state we loaded at the start.
    if hasattr(tool_handlers.get("submit", None), "_state_holder"):
        state = tool_handlers["submit"]._state_holder["state"]

    state = history.add_entry(
        state, name=name, code=code, motivation=motivation,
        bucket=bucket, flops=int(flops_max * 0.6) if flops_max else 0,
        strategy=strategy_name,
    )

    scratch_dir = scratch_dir or tempfile.mkdtemp()
    history.save_state(scratch_dir, state)
    try:
        save_scratchpad(challenge, scratch_dir)  # noqa: F821 — injected global
    except Exception as exc:
        _log(f"[agent] Scratchpad save failed: {exc}")

    return {"code": code, "name": name, "motivation": motivation}
