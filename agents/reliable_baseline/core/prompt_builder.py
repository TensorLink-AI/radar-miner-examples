"""Build rich context prompts for the LLM.

All harness parameters are read from the challenge dict, with sensible
defaults matching flops_estimator.py.  Nothing is hardcoded.
"""

from core.history import extract_flops_budget, format_history


def _get_harness_params(challenge: dict) -> tuple[int, int, int, list, int]:
    """Extract harness parameters from the challenge, with defaults."""
    task = challenge.get("task", {})
    tp = task.get("task_params", {})
    ctx = tp.get("context_len", 512)
    pred = tp.get("prediction_len", 96)
    nvar = tp.get("num_variates", 1)
    quants = tp.get("quantiles", [])
    nq = len(quants)
    return ctx, pred, nvar, quants, nq


def _compute_bucket_strategy(challenge: dict, bucket: str) -> str:
    """Generate architecture-agnostic sizing guidance from the challenge budget.

    Instead of prescribing specific architectures, computes budget-derived
    constraints the LLM can use to size whatever model it builds.
    """
    ctx, pred, nvar, quants, nq = _get_harness_params(challenge)
    flops_min, flops_max = extract_flops_budget(challenge)
    target = int(flops_max * 0.6) if flops_max else 0

    # Max hidden for a simple 2-layer channel-independent model:
    # FLOPs = V * 2 * H * (ctx + pred * nq)
    denom = 2 * nvar * (ctx + pred * nq)
    max_hidden = target // denom if denom > 0 else 0

    return (
        f"For this '{bucket}' bucket ({flops_min:,}-{flops_max:,} FLOPs), "
        f"target ~{target:,} FLOPs (60% of max). "
        f"A simple 2-layer channel-independent model can afford max hidden ~ {max_hidden}. "
        f"More complex architectures must budget FLOPs across their layers. "
        f"Choose whatever architecture best fits this budget — the estimator "
        f"measures actual forward-pass FLOPs, so any standard PyTorch ops work."
    )


def _compute_sizing_guidance(challenge: dict) -> str:
    """Build architecture-agnostic FLOPs calculator section."""
    ctx, pred, nvar, quants, nq = _get_harness_params(challenge)
    tp = challenge.get("task", {}).get("task_params", {})
    flops_min, flops_max = extract_flops_budget(challenge)
    target = int(flops_max * 0.6) if flops_max else 0
    gate_min = int(flops_min * 0.9) if flops_min else 0
    gate_max = int(flops_max * 1.1) if flops_max else 0

    denom = 2 * nvar * (ctx + pred * nq)
    max_hidden = target // denom if denom > 0 else 0

    lines = [
        "## FLOPs Calculator",
        "",
        "**FLOPs formulas for common ops:**",
        "- `nn.Linear(in, out)` on input `(batch, seq, in)` -> `2 * in * out * seq` FLOPs",
        "- `nn.Conv1d(C_in, C_out, K)` on length L -> `2 * C_in * K * C_out * L_out` FLOPs",
        "- `nn.MultiheadAttention(d, heads)` on seq S -> `8 * S * d^2 + 2 * S^2 * d` FLOPs",
        "- Note: if your model reshapes to `(batch * V, ...)`, V is already in the batch "
        "-- do NOT double-count",
        "",
        "**Budget summary:**",
        f"- Target FLOPs: {target:,} (60% of max)",
        f"- Hard gate: [{gate_min:,}, {gate_max:,}]",
        f"- context_len={ctx}, num_variates={nvar}, prediction_len={pred}, "
        f"len(quantiles)={nq}",
        "",
        "**Quick sizing:**",
        f"- A simple 2-layer model can afford max hidden ~ {max_hidden}.",
        "- ALWAYS verify your total FLOPs against the gate range.",
        "",
        "## Self-Sizing Pattern (IMPORTANT — use this in your build_model)",
        "",
        "Your `build_model()` MUST compute layer dimensions dynamically from an embedded "
        "FLOPs budget constant. Do NOT hardcode hidden dims — derive them from the budget:",
        "",
        "```python",
        f"def build_model({', '.join(tp.keys()) if tp else '**task_params'}):",
        "    n_quantiles = len(quantiles)",
        f"    TARGET_FLOPS = {target}  # 60% of max budget",
        "",
        "    out_features = prediction_len * n_quantiles",
        "    flops_per_hidden = max(1, 2 * num_variates * (context_len + out_features))",
        "    hidden_dim = max(4, TARGET_FLOPS // flops_per_hidden)",
        "",
        "    return MyModel(context_len, prediction_len, num_variates, n_quantiles, hidden_dim)",
        "```",
        "",
        "This ensures the model adapts to any task parameters and budget automatically. "
        "For multi-layer models, divide the budget across layers accordingly.",
    ]
    return "\n".join(lines)


def build_system_prompt(challenge: dict) -> str:
    """Build system prompt with domain context and hard constraints."""
    task = challenge.get("task", {})
    tp = task.get("task_params", {})
    param_str = ", ".join(tp.keys()) if tp else "**task_params"
    ctx, pred, nvar, quants, nq = _get_harness_params(challenge)
    parts: list[str] = []

    parts.append(
        "You are an expert ML researcher designing time-series forecasting models "
        "for competitive evaluation.\n\n"
        "Your code runs inside a frozen training harness. You MUST define:\n"
        f"- def build_model({param_str}) -> nn.Module\n"
        "- def build_optimizer(model) -> Optimizer\n\n"
        f"The harness calls build_model({ctx}, {pred}, "
        f"{nvar}, {quants}).\n"
        f"Input shape: (batch, {ctx}, {nvar}). "
        f"Output shape: (batch, {pred}, {nvar}, {nq})."
    )

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
        parts.append("### Anti-Patterns to Avoid\n"
                      + "\n".join(f"- {a}" for a in anti_patterns))

    parts.append(
        "### ABSOLUTE RULES\n"
        "- Respond with a SINGLE ```python code block — nothing else\n"
        "- build_model and build_optimizer MUST be top-level functions "
        "(not inside a class)\n"
        "- Use only standard PyTorch (torch, torch.nn, torch.optim, "
        "torch.nn.functional)\n"
        "- Do NOT import subprocess, socket, or ftplib\n"
        "- Use standard nn ops (nn.Linear, nn.Conv1d, nn.TransformerEncoderLayer, "
        "nn.MultiheadAttention, nn.LayerNorm) so FLOPs counter works"
    )

    return "\n\n".join(parts)


def build_user_prompt(challenge: dict, context: dict) -> str:
    """Build the user prompt with budget, frontier, DB context, and history."""
    parts: list[str] = []

    ctx, pred, nvar, quants, nq = _get_harness_params(challenge)
    tp = challenge.get("task", {}).get("task_params", {})
    param_str = ", ".join(tp.keys()) if tp else "**task_params"
    flops_min, flops_max = extract_flops_budget(challenge)
    bucket = context.get("bucket", "unknown")
    target = context.get("target_flops", int(flops_max * 0.6))
    gate_min = int(flops_min * 0.9) if flops_min else 0
    gate_max = int(flops_max * 1.1) if flops_max else 0

    parts.append(
        f"## FLOPs Budget\n"
        f"Bucket: {bucket} [{flops_min:,} - {flops_max:,}]\n"
        f"Target: ~{target:,} FLOPs (60% of max)\n"
        f"Hard gate: [{gate_min:,}, {gate_max:,}] (10% tolerance)"
    )

    # Required interface with values from the challenge
    parts.append(
        "## Required Output Structure\n"
        "```python\n"
        "import torch\n"
        "import torch.nn as nn\n\n"
        "class YourModel(nn.Module):\n"
        f"    def __init__(self, {param_str}):\n"
        "        super().__init__()\n"
        "        # ... your architecture ...\n"
        "    def forward(self, x):\n"
        f"        # x: (batch, context_len, num_variates) = (batch, {ctx}, {nvar})\n"
        f"        # return: (batch, prediction_len, num_variates, len(quantiles)) = "
        f"(batch, {pred}, {nvar}, {nq})\n"
        "        ...\n\n"
        f"def build_model({param_str}):\n"
        f"    return YourModel({param_str})\n\n"
        "def build_optimizer(model):\n"
        "    return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)\n"
        "```\n\n"
        f"IMPORTANT: num_variates={nvar}, len(quantiles)={nq}.\n"
        f"quantiles = {quants}"
    )

    # Dynamic FLOPs calculator
    parts.append(_compute_sizing_guidance(challenge))

    # Frontier context
    frontier = context.get("frontier", [])
    if frontier:
        parts.append(format_frontier(frontier, max_entries=3))
    else:
        parts.append("## Current Frontier\nNo frontier exists yet — you are bootstrapping.")

    # DB context
    db_ctx = format_db_context(
        context.get("recent_experiments", {}),
        context.get("failures", {}),
        context.get("component_stats", {}),
        context.get("dead_ends", {}),
    )
    if db_ctx != "No DB context available.":
        parts.append(f"## DB Insights\n{db_ctx}")

    # History
    history_entries = context.get("history", [])
    hist_text = format_history(history_entries, max_entries=5)
    if hist_text != "No previous submissions.":
        parts.append(f"## Your Previous Submissions\n{hist_text}")

    # Tool-assisted research findings
    tool_analysis = context.get("tool_analysis", "")
    if tool_analysis:
        parts.append(f"## Database Research Findings\n{tool_analysis}")

    # Strategy — computed from the challenge, not a hardcoded dict
    strategy = _compute_bucket_strategy(challenge, bucket)
    if strategy:
        parts.append(f"## Strategy\n{strategy}")

    parts.append(
        "## Output Format\n"
        "Return ONLY a single ```python code block with the complete code. "
        "Include optional hooks (training_config, build_scheduler, init_weights) "
        "for best training dynamics. Include RevIN (Reversible Instance Normalization) "
        "for cross-domain robustness."
    )

    return "\n\n".join(parts)


def format_frontier(frontier: list[dict], max_entries: int = 3) -> str:
    """Format frontier members for inclusion in prompt."""
    lines: list[str] = ["## Current Frontier (models you must beat)"]
    for i, member in enumerate(frontier[:max_entries]):
        metrics = member.get("objectives", {})
        lines.append(
            f"**Member {i + 1}**: "
            f"crps={metrics.get('crps', '?')}, "
            f"mase={metrics.get('mase', '?')}, "
            f"exec_time={metrics.get('exec_time', '?')}s, "
            f"memory={metrics.get('memory_mb', '?')}MB"
        )
        code = member.get("code", "")
        if code:
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
                    f"crps={m.get('crps', '?')}, flops={exp.get('flops', '?')}")

    if failures:
        items = failures if isinstance(failures, list) else failures.get("failures", [])
        if items:
            parts.append("**Recent failures:**")
            for f in items[:3]:
                parts.append(f"- {f.get('name', '?')}: {f.get('reason', '?')}")

    if component_stats:
        items = (component_stats if isinstance(component_stats, list)
                 else component_stats.get("components", []))
        if items:
            parts.append("**Component correlations:**")
            for c in (items[:5] if isinstance(items, list) else []):
                parts.append(
                    f"- {c.get('name', '?')}: "
                    f"success_rate={c.get('success_rate', '?')}")

    if dead_ends:
        items = (dead_ends if isinstance(dead_ends, list)
                 else dead_ends.get("dead_ends", []))
        if items:
            parts.append("**Dead ends:**")
            for d in (items[:3] if isinstance(items, list) else []):
                parts.append(
                    f"- {d.get('pattern', '?')}: {d.get('reason', '?')}")

    return "\n".join(parts) if parts else "No DB context available."
