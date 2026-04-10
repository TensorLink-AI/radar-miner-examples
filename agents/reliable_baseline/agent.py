"""Reliable Baseline — LLM-driven architecture design with self-sizing models.

Pipeline:
  1. Gather context (DB queries, frontier analysis)
  2. Call the LLM with rich context + self-sizing guidance
  3. Validate LLM output -> if valid, use it
  4. If LLM fails or returns invalid code -> retry ONCE with error feedback
  5. If retry fails -> skip submission (better than wrong-sized model)
"""

import sys
import tempfile

from core import llm, db_client, validation, history, tools
from core.templates import generate_fallback_code


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def design_architecture(challenge: dict, client) -> dict:
    """Entry point called by the harness.  Always returns valid code."""

    # ── STEP 0: Identify bucket and prepare fallback ──────────────
    flops_min, flops_max = history.extract_flops_budget(challenge)

    bucket = history.identify_bucket(flops_min, flops_max)
    target_flops = int(flops_max * 0.6) if flops_max else 0

    _log(f"[agent] Bucket: {bucket}, FLOPs: {flops_min:,}-{flops_max:,}, "
         f"target: {target_flops:,}")

    # ── STEP 1: Load scratchpad state ─────────────────────────────
    scratch_dir = None
    try:
        scratch_dir = load_scratchpad(challenge)  # noqa: F821 — injected global
    except Exception as exc:
        _log(f"[agent] scratchpad load failed: {exc}")

    state = history.load_state(scratch_dir) if scratch_dir else {}
    _log(f"[agent] Scratchpad loaded: {len(state)} keys")

    # ── STEP 2: Gather context (all calls try/except, failures OK) ──
    frontier = challenge.get("feasible_frontier", [])
    if not frontier:
        frontier = challenge.get("pareto_frontier", [])
    if not isinstance(frontier, list):
        frontier = []

    db_url = challenge.get("db_url", "")
    recent = db_client.recent_experiments(client, db_url) if db_url else {}
    failures = db_client.recent_failures(client, db_url) if db_url else {}
    comp_stats = db_client.component_stats(client, db_url) if db_url else {}
    dead = db_client.dead_ends(client, db_url) if db_url else {}

    context = {
        "frontier": frontier,
        "recent_experiments": recent,
        "failures": failures,
        "component_stats": comp_stats,
        "dead_ends": dead,
        "history": history.get_history(state),
        "bucket": bucket,
        "target_flops": target_flops,
    }

    # ── STEP 2b: Tool-assisted analysis (optional, best-effort) ──
    llm_url = challenge.get("llm_url", "")
    tool_analysis = ""
    if db_url and llm_url:
        try:
            tool_defs = tools.TOOLS
            tool_handlers = tools.build_handlers(client, db_url)
            analysis_messages = [
                {"role": "system", "content": (
                    "You are a research assistant. Use the provided tools to "
                    "gather information about past experiments, then summarize "
                    "what architectural patterns work well and what to avoid "
                    f"for the '{bucket}' FLOPs bucket. Be concise."
                )},
                {"role": "user", "content": (
                    "Check recent experiments, component stats, and dead ends. "
                    "Summarize findings for code generation."
                )},
            ]
            tool_analysis = llm.chat_with_tools(
                client, llm_url, analysis_messages,
                tools=tool_defs, tool_handlers=tool_handlers,
                temperature=0.3, max_rounds=4,
            )
            _log(f"[agent] Tool analysis: {len(tool_analysis)} chars")
        except Exception as exc:
            _log(f"[agent] Tool analysis failed (non-fatal): {exc}")

    if tool_analysis:
        context["tool_analysis"] = tool_analysis

    # ── STEP 3: LLM reasoning (primary path) ─────────────────────
    result = None

    if llm_url:
        try:
            result = llm.reason_and_generate(client, challenge, context)
        except Exception as exc:
            _log(f"[agent] LLM reasoning failed: {exc}")

    if result:
        code, name, motivation = result
        _log(f"[agent] LLM produced valid code: {name}")
    else:
        # LLM failed or unavailable — use dynamic fallback
        _log(f"[agent] LLM failed, generating fallback for {bucket}")
        code = generate_fallback_code(challenge)
        if code:
            ok, errors = validation.validate_code(code, challenge)
            if ok:
                name = f"fallback_{bucket}"
                motivation = "Dynamic fallback — LLM unavailable or returned invalid code"
                _log(f"[agent] Fallback code passed validation: {name}")
            else:
                _log(f"[agent] Fallback validation failed: {errors}")
                code = ""
                name = f"skipped_{bucket}"
                motivation = f"Both LLM and fallback failed: {errors}"
        else:
            name = f"skipped_{bucket}"
            motivation = "LLM unavailable and fallback not possible for this task"
            _log(f"[agent] No fallback available for {bucket}")

    # ── STEP 4: Final validation ──────────────────────────────────
    if code:
        ok, errors = validation.validate_code(code, challenge)
        if not ok:
            _log(f"[agent] Final validation failed: {errors}")
            code = ""
            name = f"skipped_{bucket}"
            motivation = f"Code failed validation: {errors}"

    # ── STEP 5: Update scratchpad ─────────────────────────────────
    state = history.add_entry(
        state, name=name, code=code, motivation=motivation,
        bucket=bucket, flops=target_flops, strategy="reliable_baseline",
    )
    scratch_dir = scratch_dir or tempfile.mkdtemp()
    history.save_state(scratch_dir, state)
    try:
        save_scratchpad(challenge, scratch_dir)  # noqa: F821 — injected global
    except Exception as exc:
        _log(f"[agent] scratchpad save failed: {exc}")

    return {"code": code, "name": name, "motivation": motivation}
