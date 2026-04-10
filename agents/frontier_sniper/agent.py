"""Frontier Sniper — surgical micro-improvements to beat the frontier by tiny margins."""

import sys
import tempfile

from core import llm, db_client, validation, prompt_builder, history, tools

STRATEGY_PREAMBLE = """You are a code reviewer, not an architect. You receive working code that is \
already competitive. Your job is to find the single most impactful improvement — a better learning \
rate schedule, a missing normalization, a suboptimal initialization — and make that one change. \
Do NOT redesign the architecture. Minimal diff, maximum impact.

Key rules:
- Start from the BEST frontier member's code — copy it almost entirely
- Change at most 1-2 things: optimizer config, a normalization layer, init scheme, LR schedule
- The sigmoid scoring has steepness=20: even 1-2% improvement on the primary metric gives ~0.55-0.65 score
- With softmax temperature=0.1, even a tiny score lead dominates the round
- NEVER change the model architecture dramatically — only tune the training dynamics
- Keep FLOPs within budget — do NOT add layers or increase hidden dims"""

def _make_bootstrap_instructions(challenge: dict, flops_max: int) -> str:
    """Build budget-aware bootstrap instructions from the challenge."""
    task = challenge.get("task", {})
    tp = task.get("task_params", {})
    target = int(flops_max * 0.6)

    lines = [
        f"No frontier exists yet. Submit a strong, proven baseline:",
        f"- Target {target:,} FLOPs (60% of max).",
    ]
    if tp:
        param_summary = ", ".join(f"{k}={v}" for k, v in tp.items())
        lines.append(f"- Task parameters: {param_summary}. Size your model accordingly.")
    lines += [
        f"- Use standard best practices: LayerNorm, residual connections, cosine LR schedule",
        f"- Use the FLOPs formulas in the calculator section to self-check your design",
        f"- Focus on reliability over novelty — be the baseline others must beat",
    ]
    return "\n".join(lines)


def get_frontier_for_bucket(challenge: dict) -> list[dict]:
    """Extract frontier members within the current bucket's FLOPs range."""
    frontier = challenge.get("feasible_frontier", [])
    if not frontier:
        frontier = challenge.get("pareto_frontier", [])
    return frontier if isinstance(frontier, list) else []


def analyze_frontier(frontier: list[dict]) -> str:
    """Build targeted analysis of frontier members for the LLM."""
    if not frontier:
        return ""

    lines = ["Analyze these frontier members and find the weakest point to improve:\n"]
    for i, member in enumerate(frontier):
        metrics = member.get("objectives", {})
        code = member.get("code", "")
        lines.append(f"--- Frontier Member {i + 1} ---")
        for metric_name, metric_val in metrics.items():
            lines.append(f"{metric_name}: {metric_val}")
        lines.append(f"FLOPs: {metrics.get('flops_equivalent_size', '?')}")
        if code:
            if len(code) > 5000:
                code = code[:5000] + "\n# ... truncated"
            lines.append(f"Code:\n```python\n{code}\n```")
        lines.append("")

    lines.append(
        "Pick the BEST member (best primary metric). Copy its code almost entirely. "
        "Make ONE surgical improvement: better LR schedule, weight init, "
        "gradient clipping, normalization, or optimizer hyperparameters. "
        "Explain your single change in a code comment."
    )
    return "\n".join(lines)


def build_strategy_instructions(frontier: list[dict], state: dict,
                                bucket: str, challenge: dict | None = None,
                                flops_max: int = 0) -> str:
    """Build strategy-specific instructions for the user prompt."""
    bucket_history = history.get_bucket_history(state, bucket)
    hist_text = history.format_history(bucket_history, max_entries=5)

    # Check scratchpad for bucket-specific playbook
    playbook = state.get("playbooks", {}).get(bucket, "")

    parts = []
    if frontier:
        parts.append(
            "STRATEGY: You are sniping the frontier. Your goal is to BARELY beat the "
            "best frontier primary metric. Copy the best frontier code and make ONE targeted change."
        )
        parts.append(analyze_frontier(frontier))
    else:
        parts.append(_make_bootstrap_instructions(challenge or {}, flops_max))

    if playbook:
        parts.append(f"### Bucket Playbook (from previous rounds)\n{playbook}")

    if hist_text != "No previous submissions.":
        parts.append(f"### Your Previous Attempts\n{hist_text}")

    return "\n\n".join(parts)


def update_playbook(state: dict, bucket: str, name: str, motivation: str) -> dict:
    """Update the per-bucket playbook in scratchpad state."""
    if "playbooks" not in state:
        state["playbooks"] = {}
    existing = state["playbooks"].get(bucket, "")
    entry = f"- {name}: {motivation}"
    if existing:
        lines = existing.strip().split("\n")
        lines.append(entry)
        # Keep last 10 entries
        state["playbooks"][bucket] = "\n".join(lines[-10:])
    else:
        state["playbooks"][bucket] = entry
    return state


def design_architecture(challenge: dict, client) -> dict:
    """Entry point called by the harness. Returns proposal dict."""
    # Identify bucket
    flops_min, flops_max = history.extract_flops_budget(challenge)
    bucket = history.identify_bucket(flops_min, flops_max)
    target_flops = int(flops_max * 0.6)

    print(f"[sniper] Bucket: {bucket}, FLOPs: {flops_min:,}-{flops_max:,}, "
          f"target: {target_flops:,}", file=sys.stderr)

    # Load scratchpad (load_scratchpad is injected by harness)
    scratch_dir = None
    try:
        scratch_dir = load_scratchpad(challenge)  # noqa: F821
    except Exception as exc:
        print(f"[sniper] scratchpad load failed: {exc}", file=sys.stderr)
    state = history.load_state(scratch_dir) if scratch_dir else {}
    print(f"[sniper] Scratchpad loaded: {len(state)} keys", file=sys.stderr)

    # Get frontier
    frontier = get_frontier_for_bucket(challenge)
    print(f"[sniper] Frontier members: {len(frontier)}", file=sys.stderr)

    # Query DB for context
    db_url = challenge.get("db_url", "")
    recent = db_client.recent_experiments(client, db_url) if db_url else {}
    failures = db_client.recent_failures(client, db_url) if db_url else {}
    comp_stats = db_client.component_stats(client, db_url) if db_url else {}
    dead = db_client.dead_ends(client, db_url) if db_url else {}

    # Build prompts
    llm_url = challenge.get("llm_url", "")
    strategy_instr = build_strategy_instructions(
        frontier, state, bucket, challenge=challenge, flops_max=flops_max,
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
                    "what surgical improvements have worked and what to avoid "
                    f"for the '{bucket}' FLOPs bucket. "
                    "Focus on training dynamics: LR schedules, optimizers, "
                    "initialization schemes, and normalization that correlate "
                    "with good primary metric scores. Be concise."
                )},
                {"role": "user", "content": (
                    "Search for experiments related to this bucket. Check "
                    "component stats and dead ends. Summarize findings."
                )},
            ]
            tool_analysis = llm.chat_with_tools(
                client, llm_url, analysis_messages,
                tools=tool_defs, tool_handlers=tool_handlers,
                temperature=0.3, max_rounds=4,
            )
            print(f"[sniper] Tool analysis: {len(tool_analysis)} chars",
                  file=sys.stderr)
        except Exception as exc:
            print(f"[sniper] Tool analysis failed (non-fatal): {exc}",
                  file=sys.stderr)

    if tool_analysis:
        user_prompt += (
            "\n\n### Database Research Findings\n" + tool_analysis
        )

    # Call LLM with validation loop (up to 3 attempts)
    code = ""
    name = "frontier_sniper_submission"
    motivation = "Surgical improvement to frontier"

    last_errors: list[str] = []

    tp = challenge.get("task", {}).get("task_params", {})
    param_str = ", ".join(tp.keys()) if tp else "**task_params"

    for attempt in range(3):
        print(f"[sniper] LLM attempt {attempt + 1}/3", file=sys.stderr)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # On retry, add correction context
        if attempt > 0:
            if code:
                messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
                ok, errors = validation.validate(code, challenge)
                messages.append({
                    "role": "user",
                    "content": (
                        f"The code has validation errors:\n"
                        + "\n".join(f"- {e}" for e in errors)
                        + "\n\nFix these errors and return the corrected code."
                    ),
                })
            else:
                # Code extraction failed — remind the LLM about the required format
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

            # Try to extract name/motivation from response
            for line in response.split("\n"):
                if line.strip().startswith("# Name:"):
                    name = line.split(":", 1)[1].strip()
                elif line.strip().startswith("# Motivation:"):
                    motivation = line.split(":", 1)[1].strip()

            ok, errors = validation.validate(code, challenge)
            if ok:
                print(f"[sniper] Validation passed on attempt {attempt + 1}",
                      file=sys.stderr)
                break
            else:
                last_errors = errors
                print(f"[sniper] Validation errors: {errors}", file=sys.stderr)
        except Exception as exc:
            last_errors = [str(exc)]
            print(f"[sniper] LLM error: {exc}", file=sys.stderr)

    # Final validation — reject invalid code instead of submitting a bad model
    ok, errors = validation.validate(code, challenge)
    if not ok:
        print(f"[sniper] Final code invalid ({errors}), skipping submission",
              file=sys.stderr)
        return {"code": "", "name": name, "motivation": f"REJECTED: {errors}"}

    # Update scratchpad
    state = history.add_entry(
        state, name=name, code=code, motivation=motivation,
        bucket=bucket, flops=target_flops, strategy="frontier_sniper",
    )
    state = update_playbook(state, bucket, name, motivation)
    scratch_dir = scratch_dir or tempfile.mkdtemp()
    history.save_state(scratch_dir, state)
    try:
        save_scratchpad(challenge, scratch_dir)  # noqa: F821
    except Exception as exc:
        print(f"[sniper] scratchpad save failed: {exc}", file=sys.stderr)

    return {
        "code": code,
        "name": name,
        "motivation": motivation,
    }
