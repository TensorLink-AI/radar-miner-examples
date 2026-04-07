"""Reliable Baseline — LLM-driven architecture design that never fails to submit.

Core philosophy: Always use the LLM for reasoning, but never let it be a single
point of failure.  A round where you submit a decent template is infinitely
better than a round where you submit nothing.

Pipeline:
  1. Prepare a guaranteed-valid template fallback BEFORE any external calls
  2. Gather context (DB queries, frontier analysis)
  3. Call the LLM with rich context — this is the primary path
  4. Validate LLM output -> if valid, use it
  5. If LLM fails or returns invalid code -> retry ONCE with error feedback
  6. If retry fails -> return the template fallback (never return empty code)
"""

import sys
import tempfile

from core import llm, db_client, validation, history, templates


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def design_architecture(challenge: dict, client) -> dict:
    """Entry point called by the harness.  Always returns valid code."""

    # ── STEP 0: Identify bucket and prepare fallback ──────────────
    flops_min = int(challenge.get("min_flops_equivalent", 0))
    flops_max = int(challenge.get("max_flops_equivalent", 0))
    # Also check nested format
    if not (flops_min or flops_max):
        fb = challenge.get("flops_budget", {})
        if isinstance(fb, dict):
            flops_min = int(fb.get("min", 0))
            flops_max = int(fb.get("max", 0))

    bucket = history.identify_bucket(flops_min, flops_max)
    target_flops = int(flops_max * 0.55) if flops_max else 0

    _log(f"[agent] Bucket: {bucket}, FLOPs: {flops_min:,}-{flops_max:,}, "
         f"target: {target_flops:,}")

    # Generate fallback FIRST — this is instant and always valid
    fallback_code = templates.get_template(bucket)
    fb_ok, fb_errors = validation.validate_code(fallback_code)
    if not fb_ok:
        _log(f"[agent] WARNING: template validation failed: {fb_errors}")
        # This should never happen — templates are unit-tested

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

    # ── STEP 3: LLM reasoning (primary path) ─────────────────────
    llm_url = challenge.get("llm_url", "")
    result = None
    source = "template_fallback"

    if llm_url:
        try:
            result = llm.reason_and_generate(client, challenge, context)
        except Exception as exc:
            _log(f"[agent] LLM reasoning failed: {exc}")

    if result:
        code, name, motivation = result
        source = "llm"
        _log(f"[agent] LLM produced valid code: {name}")
    else:
        # LLM failed or unavailable — use the template
        code = fallback_code
        name = f"template_{bucket}"
        motivation = f"LLM unavailable — reliable {bucket} baseline"
        _log(f"[agent] Using template fallback for {bucket}")

    # ── STEP 4: Final safety check (belt + suspenders) ────────────
    ok, errors = validation.validate_code(code)
    if not ok:
        _log(f"[agent] SAFETY: code from {source} failed validation: {errors}")
        _log(f"[agent] SAFETY: falling back to template")
        code = fallback_code
        name = f"template_{bucket}"
        motivation = f"Safety fallback — {source} code failed: {errors}"

    # ── STEP 5: Update scratchpad ─────────────────────────────────
    state = history.add_entry(
        state, name=name, code=code, motivation=motivation,
        bucket=bucket, flops=target_flops, strategy=source,
    )
    scratch_dir = scratch_dir or tempfile.mkdtemp()
    history.save_state(scratch_dir, state)
    try:
        save_scratchpad(challenge, scratch_dir)  # noqa: F821 — injected global
    except Exception as exc:
        _log(f"[agent] scratchpad save failed: {exc}")

    return {"code": code, "name": name, "motivation": motivation}
