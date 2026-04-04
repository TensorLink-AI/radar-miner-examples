"""Task-aware prompt construction from challenge spec."""

import json


def build_system_prompt(challenge: dict, strategy_preamble: str = "") -> str:
    """Build system prompt from challenge task fields + strategy overlay."""
    task = challenge.get("task", {})
    parts: list[str] = []

    if strategy_preamble:
        parts.append(strategy_preamble)

    domain = task.get("domain_system_prompt", "")
    if domain:
        parts.append(f"### Domain Context\n{domain}")

    constraints = task.get("constraints", [])
    if constraints:
        parts.append("### Constraints\n" + "\n".join(f"- {c}" for c in constraints))

    objectives = task.get("objectives", [])
    if objectives:
        parts.append("### Objectives\n" + "\n".join(f"- {o}" for o in objectives))

    anti_patterns = task.get("anti_patterns", [])
    if anti_patterns:
        parts.append("### Anti-Patterns to Avoid\n" + "\n".join(f"- {a}" for a in anti_patterns))

    example_hypotheses = task.get("example_hypotheses", [])
    if example_hypotheses:
        parts.append("### Example Hypotheses\n" + "\n".join(f"- {h}" for h in example_hypotheses))

    return "\n\n".join(parts)


def build_user_prompt(challenge: dict, *,
                      frontier_context: str = "",
                      db_context: str = "",
                      history_context: str = "",
                      strategy_instructions: str = "") -> str:
    """Build the user prompt with budget, frontier, DB context, and history."""
    parts: list[str] = []

    # FLOPs budget
    flops_min = challenge.get("flops_budget", {}).get("min", 0)
    flops_max = challenge.get("flops_budget", {}).get("max", 0)
    target = int(flops_max * 0.6) if flops_max else 0
    parts.append(
        f"### FLOPs Budget\n"
        f"- Min: {flops_min:,}\n"
        f"- Max: {flops_max:,}\n"
        f"- Target ~60% of max: {target:,}\n"
        f"- Hard gate: [{int(flops_min * 0.9):,}, {int(flops_max * 1.1):,}]"
    )

    # Task info
    task = challenge.get("task", {})
    run_cmd = task.get("run_command", "")
    if "harness.py" in run_cmd:
        parts.append(
            "### Required Interface (Harness Task)\n"
            "You MUST define:\n"
            "- `build_model(context_len, prediction_len, num_variates, quantiles) -> nn.Module`\n"
            "- `build_optimizer(model) -> Optimizer`\n\n"
            "Optional hooks: training_config(), init_weights(), configure_amp(), "
            "transform_batch(), on_step_end(), build_scheduler(), compute_loss()"
        )

    if strategy_instructions:
        parts.append(f"### Strategy\n{strategy_instructions}")

    if frontier_context:
        parts.append(f"### Current Frontier\n{frontier_context}")

    if db_context:
        parts.append(f"### DB Insights\n{db_context}")

    if history_context:
        parts.append(f"### History\n{history_context}")

    parts.append(
        "### Output Format\n"
        "Return ONLY a single Python code block (```python ... ```) with the complete code. "
        "No explanations outside the code block."
    )

    return "\n\n".join(parts)


def format_frontier(frontier: list[dict], max_entries: int = 3) -> str:
    """Format frontier members for inclusion in prompt."""
    if not frontier:
        return "No frontier exists yet — you are bootstrapping."

    lines: list[str] = []
    for i, member in enumerate(frontier[:max_entries]):
        metrics = member.get("metrics", {})
        lines.append(
            f"**Member {i + 1}**: "
            f"crps={metrics.get('crps', '?')}, "
            f"mase={metrics.get('mase', '?')}, "
            f"exec_time={metrics.get('exec_time', '?')}s, "
            f"memory={metrics.get('memory_mb', '?')}MB, "
            f"flops={member.get('flops_equivalent_size', '?')}"
        )
        code = member.get("code", "")
        if code:
            # Truncate very long code
            if len(code) > 4000:
                code = code[:4000] + "\n# ... (truncated)"
            lines.append(f"```python\n{code}\n```")

    return "\n\n".join(lines)


def format_db_context(recent: dict | list, failures: dict | list,
                      component_stats: dict | list,
                      dead_ends: dict | list) -> str:
    """Format DB query results into a compact context string."""
    parts: list[str] = []

    if recent:
        items = recent if isinstance(recent, list) else recent.get("experiments", [])
        if items:
            parts.append("**Recent experiments:**")
            for exp in items[:5]:
                m = exp.get("metrics", {})
                parts.append(
                    f"- {exp.get('name', '?')}: "
                    f"crps={m.get('crps', '?')}, flops={exp.get('flops', '?')}"
                )

    if failures:
        items = failures if isinstance(failures, list) else failures.get("failures", [])
        if items:
            parts.append("**Recent failures:**")
            for f in items[:3]:
                parts.append(f"- {f.get('name', '?')}: {f.get('reason', '?')}")

    if component_stats:
        items = component_stats if isinstance(component_stats, list) else component_stats.get("components", [])
        if items:
            parts.append("**Component correlations:**")
            for c in (items[:5] if isinstance(items, list) else []):
                parts.append(f"- {c.get('name', '?')}: success_rate={c.get('success_rate', '?')}")

    if dead_ends:
        items = dead_ends if isinstance(dead_ends, list) else dead_ends.get("dead_ends", [])
        if items:
            parts.append("**Dead ends:**")
            for d in (items[:3] if isinstance(items, list) else []):
                parts.append(f"- {d.get('pattern', '?')}: {d.get('reason', '?')}")

    return "\n".join(parts) if parts else "No DB context available."
