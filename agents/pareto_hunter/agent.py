"""Pareto Hunter — exploit the 1.5x dominance bonus by targeting all objectives."""

import sys
import tempfile

from core import llm, db_client, validation, prompt_builder, history, tools

STRATEGY_PREAMBLE = """You are a multi-objective optimizer. You MUST beat the frontier on ALL \
metrics simultaneously to earn the 1.5x dominance bonus. The task's objectives \
(primary and secondary) are listed in the Objectives section below — read them carefully.

Most miners only optimize the primary metric. The secondary metrics are your ATTACK SURFACE. \
A model that matches the frontier's primary metric while being significantly better on \
secondary objectives will DOMINATE and earn the 1.5x multiplier.

Design principles for multi-objective dominance:
- Use bfloat16 via configure_amp() for memory savings
- Use larger batch sizes via training_config() for faster wall-clock
- Use init_weights() for faster convergence (good primary metric in fewer steps)
- Use compute_loss() to jointly optimize the primary and secondary objectives
- Use the self-sizing pattern and FLOPs formulas to fit within budget efficiently"""



def analyze_frontier_weaknesses(frontier: list[dict]) -> str:
    """Identify which frontier members are 'dominatable' on secondary objectives."""
    if not frontier:
        return "No frontier — design for efficiency from the start."

    lines = ["### Frontier Weakness Analysis\n"]
    for i, member in enumerate(frontier):
        metrics = member.get("objectives", {})

        lines.append(f"**Member {i + 1}:**")
        metric_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
        lines.append(f"  Objectives: {metric_str}")
        lines.append(
            "  Analyze these metrics to find weaknesses you can exploit for Pareto dominance."
        )

        code = member.get("code", "")
        if code:
            if len(code) > 4000:
                code = code[:4000] + "\n# ... truncated"
            lines.append(f"  Code:\n```python\n{code}\n```")
        lines.append("")

    return "\n".join(lines)


def get_dominatable_targets(frontier: list[dict]) -> list[dict]:
    """Return all frontier members as potential domination targets.

    Rather than applying hardcoded thresholds, we return every member
    and let the LLM reason about which are weakest on secondary metrics.
    """
    return list(frontier)


def build_strategy_instructions(frontier: list[dict], state: dict,
                                bucket: str) -> str:
    """Build Pareto-hunter strategy instructions."""
    parts = []

    if frontier:
        weakness_analysis = analyze_frontier_weaknesses(frontier)
        parts.append(weakness_analysis)

        targets = get_dominatable_targets(frontier)
        if targets:
            parts.append(
                f"Found {len(targets)} DOMINATABLE frontier members. "
                "Design an architecture that beats ALL their metrics. "
                "The 1.5x bonus is your primary advantage."
            )
        else:
            parts.append(
                "No obviously dominatable members — focus on matching the best "
                "primary metric while being extremely efficient on secondary "
                "objectives. Use bfloat16, large batches, simple but effective "
                "architectures."
            )

        # Track dominatable targets in state
        state.setdefault("dominatable", {})[bucket] = len(targets)
    else:
        parts.append(
            "No frontier yet. Design an EFFICIENT baseline from the start:\n"
            "- Use bfloat16 (configure_amp)\n"
            "- Large batch size (training_config)\n"
            "- Choose an architecture that fits the FLOPs budget efficiently\n"
            "- Proper init (init_weights) for fast convergence\n"
            "This establishes a Pareto-dominant position early."
        )

    # Previous Pareto attempts
    pareto_hist = [
        e for e in history.get_history(state)
        if e.get("strategy") == "pareto_hunter"
    ]
    if pareto_hist:
        parts.append(
            "### Previous Pareto Attempts\n"
            + history.format_history(pareto_hist, max_entries=5)
        )

    return "\n\n".join(parts)


def design_architecture(challenge: dict, client) -> dict:
    """Entry point called by the harness. Returns proposal dict."""
    flops_min, flops_max = history.extract_flops_budget(challenge)
    bucket = history.identify_bucket(flops_min, flops_max)
    target_flops = int(flops_max * 0.6)

    print(f"[pareto] Bucket: {bucket}, FLOPs: {flops_min:,}-{flops_max:,}, "
          f"target: {target_flops:,}", file=sys.stderr)

    # Load scratchpad state (load_scratchpad is injected by harness)
    scratch_dir = None
    try:
        scratch_dir = load_scratchpad(challenge)  # noqa: F821
    except Exception as exc:
        print(f"[pareto] scratchpad load failed: {exc}", file=sys.stderr)
    state = history.load_state(scratch_dir) if scratch_dir else {}
    print(f"[pareto] Scratchpad loaded: {len(state)} keys", file=sys.stderr)

    frontier = challenge.get("feasible_frontier", [])
    if not frontier:
        frontier = challenge.get("pareto_frontier", [])
    if not isinstance(frontier, list):
        frontier = []
    print(f"[pareto] Frontier members: {len(frontier)}", file=sys.stderr)

    # Query DB
    db_url = challenge.get("db_url", "")
    recent = db_client.recent_experiments(client, db_url) if db_url else {}
    failures = db_client.recent_failures(client, db_url) if db_url else {}
    comp_stats = db_client.component_stats(client, db_url) if db_url else {}
    dead = db_client.dead_ends(client, db_url) if db_url else {}

    # Build prompts
    llm_url = challenge.get("llm_url", "")
    strategy_instr = build_strategy_instructions(frontier, state, bucket)
    frontier_ctx = prompt_builder.format_frontier(frontier, max_entries=5)
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

    # Add specific efficiency instructions to user prompt
    efficiency_addendum = (
        "\n\n### Efficiency Requirements\n"
        "Your code MUST include these hooks for multi-objective dominance:\n"
        "1. `configure_amp()` returning `{'enabled': True, 'dtype': 'bfloat16'}` for memory savings\n"
        "2. `training_config()` with larger batch_size for faster wall-clock\n"
        "3. `init_weights(model)` with proper initialization for fast convergence\n"
        "4. Consider `compute_loss()` to jointly optimize the task's primary and secondary objectives\n"
    )
    user_prompt += efficiency_addendum

    # --- Tool-assisted analysis phase (optional, best-effort) ---
    tool_analysis = ""
    if db_url and llm_url:
        try:
            tool_defs = tools.TOOLS
            tool_handlers = tools.build_handlers(client, db_url)
            analysis_messages = [
                {"role": "system", "content": (
                    "You are a research assistant. Use the provided tools to "
                    "gather information about past experiments, then summarize "
                    "what approaches achieve good multi-objective performance "
                    f"in the '{bucket}' FLOPs bucket. "
                    "Focus on all objectives, not just the primary metric. "
                    "Identify which components are efficient. Be concise."
                )},
                {"role": "user", "content": (
                    "Search for efficient experiments. Check component stats "
                    "and dead ends. Summarize what achieves Pareto dominance."
                )},
            ]
            tool_analysis = llm.chat_with_tools(
                client, llm_url, analysis_messages,
                tools=tool_defs, tool_handlers=tool_handlers,
                temperature=0.3, max_rounds=4,
            )
            print(f"[pareto] Tool analysis: {len(tool_analysis)} chars",
                  file=sys.stderr)
        except Exception as exc:
            print(f"[pareto] Tool analysis failed (non-fatal): {exc}",
                  file=sys.stderr)

    if tool_analysis:
        user_prompt += (
            "\n\n### Database Research Findings\n" + tool_analysis
        )

    # LLM call with validation loop
    code = ""
    name = f"pareto_hunter_{bucket}"
    motivation = "Multi-objective dominance targeting 1.5x Pareto bonus"
    last_errors: list[str] = []

    tp = challenge.get("task", {}).get("task_params", {})
    param_str = ", ".join(tp.keys()) if tp else "**task_params"

    for attempt in range(3):
        print(f"[pareto] LLM attempt {attempt + 1}/3", file=sys.stderr)

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
                        + "\n\nFix these errors. Return corrected code. "
                        "Remember: include configure_amp, training_config, and init_weights."
                    ),
                })
            else:
                error_detail = "; ".join(last_errors) if last_errors else "no code block found"
                messages.append({
                    "role": "user",
                    "content": (
                        f"Previous attempt failed: {error_detail}. "
                        "You MUST respond with a single ```python code block containing "
                        f"def build_model({param_str}) "
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
                print(f"[pareto] Validation passed on attempt {attempt + 1}",
                      file=sys.stderr)
                break
            else:
                last_errors = errors
                print(f"[pareto] Validation errors: {errors}", file=sys.stderr)
        except Exception as exc:
            last_errors = [str(exc)]
            print(f"[pareto] LLM error: {exc}", file=sys.stderr)

    ok, errors = validation.validate(code, challenge)
    if not ok:
        print(f"[pareto] Final code invalid ({errors}), skipping submission",
              file=sys.stderr)
        return {"code": "", "name": name, "motivation": f"REJECTED: {errors}"}

    # Update scratchpad
    state = history.add_entry(
        state, name=name, code=code, motivation=motivation,
        bucket=bucket, flops=target_flops, strategy="pareto_hunter",
    )
    scratch_dir = scratch_dir or tempfile.mkdtemp()
    history.save_state(scratch_dir, state)
    try:
        save_scratchpad(challenge, scratch_dir)  # noqa: F821
    except Exception as exc:
        print(f"[pareto] scratchpad save failed: {exc}", file=sys.stderr)

    return {
        "code": code,
        "name": name,
        "motivation": motivation,
    }
