"""Simple Modeler — focus on building a good model, no overthinking."""

import sys
import tempfile

from core import llm, db_client, validation, prompt_builder, history, tools

STRATEGY_PREAMBLE = """\
You are a pragmatic ML engineer. Your goal is simple: build a well-designed PyTorch model \
for time-series forecasting that fits within the given FLOPs budget and produces good predictions.

Do NOT overthink strategy. Do NOT try to game scoring. Just build a solid model:
- Pick an appropriate architecture for the FLOPs budget
- Use standard best practices (LayerNorm, residual connections, proper init)
- Set reasonable training hyperparameters (learning rate, batch size, epochs)
- Make sure the output shape is exactly right
- Keep it clean and correct — a working simple model beats a broken clever one"""


def get_frontier(challenge: dict) -> list[dict]:
    """Extract frontier members from challenge."""
    frontier = challenge.get("feasible_frontier", [])
    if not frontier:
        frontier = challenge.get("pareto_frontier", [])
    return frontier if isinstance(frontier, list) else []


def build_strategy_instructions(frontier: list[dict], state: dict,
                                bucket: str, flops_min: int,
                                flops_max: int) -> str:
    """Build simple, direct instructions — no over-strategising."""
    target = int(flops_max * 0.6)
    parts = []

    parts.append(
        f"Build a good model for the '{bucket}' bucket "
        f"({flops_min:,}-{flops_max:,} FLOPs). "
        f"Target around {target:,} FLOPs."
    )

    # Simple size-based guidance
    if bucket == "tiny":
        parts.append(
            "This is a very small budget. Use a lightweight model: "
            "linear layers, small MLPs, or simple conv layers. "
            "Avoid attention — it's too expensive here."
        )
    elif bucket == "small":
        parts.append(
            "Small budget. A 1-2 layer model with small hidden dims (32-64) works well. "
            "Linear attention or conv+MLP mixers are good choices."
        )
    elif bucket in ("medium_small", "medium"):
        parts.append(
            "Medium budget. You can use a multi-layer transformer with attention, "
            "residual connections, and normalization. 2-6 layers, hidden dim 64-256."
        )
    elif bucket == "large":
        parts.append(
            "Large budget. Full transformer architectures work well. "
            "6-12 layers, hidden dim 256-512, multi-head attention. "
            "Focus on training dynamics (LR schedule, warmup)."
        )

    if frontier:
        # Just show what exists — don't tell the LLM to game anything
        best = min(frontier, key=lambda m: m.get("objectives", {}).get("crps", float("inf")))
        best_crps = best.get("objectives", {}).get("crps", "?")
        parts.append(
            f"The current best CRPS on the frontier is {best_crps}. "
            "Study the frontier code for inspiration, but don't just copy it — "
            "build something that works well."
        )
    else:
        parts.append(
            "No frontier exists yet. Submit a strong, well-tested baseline."
        )

    # Show recent history to avoid repeating failures
    bucket_hist = history.get_bucket_history(state, bucket)
    if bucket_hist:
        parts.append(
            f"### Your Recent Attempts for '{bucket}'\n"
            + history.format_history(bucket_hist, max_entries=5)
        )

    return "\n\n".join(parts)


def design_architecture(challenge: dict, client) -> dict:
    """Entry point called by the harness. Returns proposal dict."""
    flops_min, flops_max = history.extract_flops_budget(challenge)
    bucket = history.identify_bucket(flops_min, flops_max)
    target_flops = int(flops_max * 0.6)

    print(f"[simple] Bucket: {bucket}, FLOPs: {flops_min:,}-{flops_max:,}, "
          f"target: {target_flops:,}", file=sys.stderr)

    # Load scratchpad
    scratch_dir = load_scratchpad(challenge)  # noqa: F821
    state = history.load_state(scratch_dir) if scratch_dir else {}

    # Get frontier
    frontier = get_frontier(challenge)
    print(f"[simple] Frontier members: {len(frontier)}", file=sys.stderr)

    # Query DB for context
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
                    "what architectural patterns work well and what to avoid "
                    f"for the '{bucket}' FLOPs bucket "
                    f"({flops_min:,}-{flops_max:,} FLOPs). "
                    "Be concise — your summary will feed into code generation."
                )},
                {"role": "user", "content": (
                    "Search the experiment database for relevant information. "
                    "Check component stats, dead ends, and recent experiments. "
                    "Then write a brief summary of findings."
                )},
            ]
            tool_analysis = llm.chat_with_tools(
                client, llm_url, analysis_messages,
                tools=tool_defs, tool_handlers=tool_handlers,
                temperature=0.3, max_rounds=4,
            )
            print(f"[simple] Tool analysis: {len(tool_analysis)} chars",
                  file=sys.stderr)
        except Exception as exc:
            print(f"[simple] Tool analysis failed (non-fatal): {exc}",
                  file=sys.stderr)

    # Inject tool analysis into user prompt if available
    if tool_analysis:
        user_prompt += (
            "\n\n### Database Research Findings\n" + tool_analysis
        )

    # LLM call with validation loop (up to 3 attempts)
    code = ""
    last_errors: list[str] = []
    name = f"simple_modeler_{bucket}"
    motivation = f"Well-designed model for {bucket} bucket"

    for attempt in range(3):
        print(f"[simple] LLM attempt {attempt + 1}/3", file=sys.stderr)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # On retry, feed back correction context
        if attempt > 0:
            if code:
                messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
                messages.append({
                    "role": "user",
                    "content": (
                        "The code has validation errors:\n"
                        + "\n".join(f"- {e}" for e in last_errors)
                        + "\n\nFix these errors and return the corrected code."
                    ),
                })
            else:
                # Empty code — LLM didn't return a fenced code block
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response did not contain a fenced Python code block. "
                        "You MUST respond with a single ```python ... ``` block containing "
                        "top-level `def build_model(context_len, prediction_len, num_variates, quantiles)` "
                        "and `def build_optimizer(model)` functions. Try again."
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
            last_errors = errors
            if ok:
                print(f"[simple] Validation passed on attempt {attempt + 1}",
                      file=sys.stderr)
                break
            else:
                print(f"[simple] Validation errors: {errors}", file=sys.stderr)
        except Exception as exc:
            print(f"[simple] LLM error: {exc}", file=sys.stderr)

    # Final validation gate
    ok, errors = validation.validate(code, challenge)
    if not ok:
        print(f"[simple] Final code invalid ({errors}), skipping submission",
              file=sys.stderr)
        return {"code": "", "name": name, "motivation": f"REJECTED: {errors}"}

    # Update scratchpad
    state = history.add_entry(
        state, name=name, code=code, motivation=motivation,
        bucket=bucket, flops=target_flops, strategy="simple_modeler",
    )
    scratch_dir = scratch_dir or tempfile.mkdtemp()
    history.save_state(scratch_dir, state)
    save_scratchpad(challenge, scratch_dir)  # noqa: F821

    return {
        "code": code,
        "name": name,
        "motivation": motivation,
    }
