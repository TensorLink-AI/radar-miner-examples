"""Bucket Specialist — dominate specific FLOPs ranges with pre-optimized templates."""

import sys
import tempfile

from core import llm, db_client, validation, prompt_builder, history

STRATEGY_PREAMBLE = """You are optimizing for a specific compute budget. You have deep expertise \
with models in this exact size range. You maintain a library of bucket-specific architectures \
that have been refined over many rounds.

Key rules:
- Each size bucket has different optimal architectures. Tiny models need different designs than large ones.
- Target exactly 60% of the bucket's max FLOPs for safety margin within the 10% tolerance gate.
- You have specialized templates for each bucket — use and evolve them, don't start from scratch.
- EMA alpha=0.3 means winning 2-3 out of 5 rounds sustains strong weight over time.
- If you're already on the frontier for this bucket, optimize secondary objectives for the 1.5x Pareto bonus.
- Track what works per-bucket obsessively — this is your competitive advantage."""

BUCKET_TEMPLATES = {
    "tiny": {
        "description": "100K-500K FLOPs. Use lightweight linear mixers, small MLPs, or patch-based models.",
        "tips": [
            "Avoid attention — too expensive for this range",
            "Depthwise separable convolutions are efficient here",
            "Single-layer models with good feature engineering",
            "Patch embeddings to reduce sequence length",
        ],
    },
    "small": {
        "description": "500K-2M FLOPs. Small transformers or efficient conv models.",
        "tips": [
            "1-2 layer transformers with small hidden dim (32-64)",
            "Linear attention variants",
            "Conv1d + MLP mixer architectures",
            "Careful: self-attention FLOPs scale quadratically",
        ],
    },
    "medium_small": {
        "description": "2M-10M FLOPs. Mid-size models with room for architectural tricks.",
        "tips": [
            "2-4 layer transformers with hidden dim 64-128",
            "Can afford multi-head attention (2-4 heads)",
            "Residual connections become important here",
            "Consider patch-based approaches to reduce sequence length",
        ],
    },
    "medium": {
        "description": "10M-50M FLOPs. Full architectures possible.",
        "tips": [
            "4-8 layer transformers, hidden dim 128-256",
            "Multi-head attention with 4-8 heads",
            "Can include normalization layers and dropout",
            "Consider mixture-of-experts for efficiency",
        ],
    },
    "large": {
        "description": "50M-125M FLOPs. Complex architectures viable.",
        "tips": [
            "6-12 layer transformers, hidden dim 256-512",
            "Full attention mechanisms affordable",
            "Can stack more sophisticated blocks",
            "Focus on training dynamics (LR schedule, warmup) over architecture size",
        ],
    },
}


def get_bucket_template_prompt(bucket: str, state: dict) -> str:
    """Get the bucket-specific template info and any saved templates."""
    parts = []

    # Static template guidance
    template = BUCKET_TEMPLATES.get(bucket, {})
    if template:
        parts.append(f"**Bucket Profile**: {template.get('description', '')}")
        tips = template.get("tips", [])
        if tips:
            parts.append("**Design Tips:**\n" + "\n".join(f"- {t}" for t in tips))

    # Saved template from previous rounds
    saved = state.get("templates", {}).get(bucket, "")
    if saved:
        parts.append(f"**Your Best Previous Code for This Bucket:**\n```python\n{saved}\n```")
        metrics = state.get("template_metrics", {}).get(bucket, {})
        if metrics:
            parts.append(
                f"Previous results: crps={metrics.get('crps', '?')}, "
                f"flops={metrics.get('flops', '?')}"
            )

    return "\n\n".join(parts)


def build_strategy_instructions(frontier: list[dict], state: dict,
                                bucket: str, flops_min: int,
                                flops_max: int) -> str:
    """Build bucket-specialist strategy instructions."""
    parts = []

    target = int(flops_max * 0.6)
    parts.append(
        f"STRATEGY: You are a SPECIALIST for the '{bucket}' bucket "
        f"({flops_min:,}-{flops_max:,} FLOPs). "
        f"Target exactly {target:,} FLOPs."
    )

    # Bucket template
    template_info = get_bucket_template_prompt(bucket, state)
    if template_info:
        parts.append(template_info)

    # Frontier analysis for this bucket
    if frontier:
        parts.append(
            f"There are {len(frontier)} frontier members. "
            "Study them and design an architecture that beats their CRPS. "
            "If your previous template already matches frontier quality, "
            "focus on secondary objectives (exec_time, memory_mb) for the 1.5x Pareto bonus."
        )
    else:
        parts.append(
            "No frontier yet — submit your best template for this bucket to establish position."
        )

    # Bucket history
    bucket_hist = history.get_bucket_history(state, bucket)
    if bucket_hist:
        parts.append(
            f"### Your History for '{bucket}' Bucket\n"
            + history.format_history(bucket_hist, max_entries=5)
        )

    return "\n\n".join(parts)


def save_template(state: dict, bucket: str, code: str,
                  metrics: dict | None = None) -> dict:
    """Save the submitted code as the template for this bucket."""
    if "templates" not in state:
        state["templates"] = {}
    state["templates"][bucket] = code

    if metrics:
        if "template_metrics" not in state:
            state["template_metrics"] = {}
        state["template_metrics"][bucket] = metrics

    return state


def design_architecture(challenge: dict, client) -> dict:
    """Entry point called by the harness. Returns proposal dict."""
    flops_min, flops_max = history.extract_flops_budget(challenge)
    bucket = history.identify_bucket(flops_min, flops_max)
    target_flops = int(flops_max * 0.6)

    print(f"[specialist] Bucket: {bucket}, FLOPs: {flops_min:,}-{flops_max:,}, "
          f"target: {target_flops:,}", file=sys.stderr)

    # Load scratchpad (load_scratchpad is injected by harness)
    scratch_dir = load_scratchpad(challenge)  # noqa: F821
    state = history.load_state(scratch_dir) if scratch_dir else {}
    print(f"[specialist] Scratchpad loaded: {len(state)} keys", file=sys.stderr)

    frontier = challenge.get("feasible_frontier", [])
    if not isinstance(frontier, list):
        frontier = []
    print(f"[specialist] Frontier members: {len(frontier)}", file=sys.stderr)

    # Query DB — focus on bucket-relevant experiments
    db_url = challenge.get("db_url", "")
    recent = db_client.recent_experiments(client, db_url) if db_url else {}
    failures = db_client.recent_failures(client, db_url) if db_url else {}
    comp_stats = db_client.component_stats(client, db_url) if db_url else {}
    dead = db_client.dead_ends(client, db_url) if db_url else {}

    # Build prompts
    llm_url = challenge.get("llm_url", "")
    strategy_instr = build_strategy_instructions(
        frontier, state, bucket, flops_min, flops_max
    )
    frontier_ctx = prompt_builder.format_frontier(frontier, max_entries=3)
    db_ctx = prompt_builder.format_db_context(recent, failures, comp_stats, dead)
    hist_ctx = history.format_history(history.get_history(state), max_entries=5)

    system_prompt = prompt_builder.build_system_prompt(
        challenge, strategy_preamble=STRATEGY_PREAMBLE
    )
    user_prompt = prompt_builder.build_user_prompt(
        challenge,
        frontier_context=frontier_ctx,
        db_context=db_ctx,
        history_context=hist_ctx,
        strategy_instructions=strategy_instr,
    )

    # LLM call with validation loop
    code = ""
    name = f"bucket_specialist_{bucket}"
    motivation = f"Bucket-specialized architecture for {bucket}"
    last_errors: list[str] = []

    for attempt in range(3):
        print(f"[specialist] LLM attempt {attempt + 1}/3", file=sys.stderr)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if attempt > 0:
            if code:
                messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
                ok, errors = validation.validate(code, challenge)
                messages.append({
                    "role": "user",
                    "content": (
                        f"Validation errors:\n"
                        + "\n".join(f"- {e}" for e in errors)
                        + "\n\nFix these errors. Return corrected code."
                    ),
                })
            else:
                error_detail = "; ".join(last_errors) if last_errors else "no code block found"
                messages.append({
                    "role": "user",
                    "content": (
                        f"Previous attempt failed: {error_detail}. "
                        "You MUST respond with a single ```python code block containing "
                        "def build_model(context_len, prediction_len, num_variates, quantiles) "
                        "and def build_optimizer(model). No text outside the code block."
                    ),
                })

        try:
            response = llm.chat(client, llm_url, messages, temperature=0.7)
            code = llm.extract_code(response)

            for line in response.split("\n"):
                if line.strip().startswith("# Name:"):
                    name = line.split(":", 1)[1].strip()
                elif line.strip().startswith("# Motivation:"):
                    motivation = line.split(":", 1)[1].strip()

            ok, errors = validation.validate(code, challenge)
            if ok:
                print(f"[specialist] Validation passed on attempt {attempt + 1}",
                      file=sys.stderr)
                break
            else:
                last_errors = errors
                print(f"[specialist] Validation errors: {errors}", file=sys.stderr)
        except Exception as exc:
            last_errors = [str(exc)]
            print(f"[specialist] LLM error: {exc}", file=sys.stderr)

    ok, errors = validation.validate(code, challenge)
    if not ok:
        print(f"[specialist] Final code invalid ({errors}), skipping submission",
              file=sys.stderr)
        return {"code": "", "name": name, "motivation": f"REJECTED: {errors}"}

    # Update scratchpad
    state = history.add_entry(
        state, name=name, code=code, motivation=motivation,
        bucket=bucket, flops=target_flops, strategy="bucket_specialist",
    )
    state = save_template(state, bucket, code)
    scratch_dir = scratch_dir or tempfile.mkdtemp()
    history.save_state(scratch_dir, state)
    save_scratchpad(challenge, scratch_dir)  # noqa: F821

    return {
        "code": code,
        "name": name,
        "motivation": motivation,
    }
