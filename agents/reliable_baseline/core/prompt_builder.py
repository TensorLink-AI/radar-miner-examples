"""Build rich context prompts for the LLM.

CRITICAL: Hardcode correct values for num_variates=1 and
quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].
Do NOT trust challenge["task"].get() with wrong defaults.
"""

from core.history import extract_flops_budget, format_history

# Correct harness values — hardcoded, not from task.get() with wrong defaults
CONTEXT_LEN = 512
PREDICTION_LEN = 96
NUM_VARIATES = 1
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_QUANTILES = len(QUANTILES)  # 9

BUCKET_STRATEGIES = {
    "tiny": (
        "For this TINY bucket (~275K FLOPs), use a simple linear mixer or MLP. "
        "No attention — it is too expensive. Focus on: efficient linear projections, "
        "RevIN normalization, good weight init. A 2-layer MLP with small hidden dim "
        "(16-32) works well."
    ),
    "small": (
        "For this SMALL bucket (~1.1M FLOPs), use lightweight conv + linear. "
        "A 1D conv for local patterns plus a linear mixer for global patterns. "
        "Hidden dim 32-48, kernel size 5-9. Include RevIN."
    ),
    "medium_small": (
        "For this MEDIUM_SMALL bucket (~5.5M FLOPs), PatchTST-style architecture "
        "works well: patch the input into segments, apply a small transformer "
        "(2-3 layers, d_model=64, 4 heads). Include RevIN and cosine LR warmup."
    ),
    "medium": (
        "For this MEDIUM bucket (~27M FLOPs), use a multi-layer transformer with "
        "patching (3-4 layers, d_model=128, 4 heads, ff=256). Include RevIN, "
        "warmup + cosine schedule, and Xavier init."
    ),
    "large": (
        "For this LARGE bucket (~69M FLOPs), use a full transformer (4-6 layers, "
        "d_model=192-256, 8 heads). Include RevIN, warmup + cosine schedule, "
        "gradient clipping, and Xavier init. Consider adding a projection head."
    ),
}


def build_system_prompt(challenge: dict) -> str:
    """Build system prompt with domain context and hard constraints."""
    task = challenge.get("task", {})
    parts: list[str] = []

    parts.append(
        "You are an expert ML researcher designing time-series forecasting models "
        "for competitive evaluation.\n\n"
        "Your code runs inside a frozen training harness. You MUST define:\n"
        "- def build_model(context_len, prediction_len, num_variates, quantiles) -> nn.Module\n"
        "- def build_optimizer(model) -> Optimizer\n\n"
        f"The harness calls build_model({CONTEXT_LEN}, {PREDICTION_LEN}, "
        f"{NUM_VARIATES}, {QUANTILES}).\n"
        f"Input shape: (batch, {CONTEXT_LEN}, {NUM_VARIATES}). "
        f"Output shape: (batch, {PREDICTION_LEN}, {NUM_VARIATES}, {N_QUANTILES})."
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

    flops_min, flops_max = extract_flops_budget(challenge)
    bucket = context.get("bucket", "unknown")
    target = context.get("target_flops", int(flops_max * 0.55))
    gate_min = int(flops_min * 0.9) if flops_min else 0
    gate_max = int(flops_max * 1.1) if flops_max else 0

    parts.append(
        f"## FLOPs Budget\n"
        f"Bucket: {bucket} [{flops_min:,} - {flops_max:,}]\n"
        f"Target: ~{target:,} FLOPs (55% of max)\n"
        f"Hard gate: [{gate_min:,}, {gate_max:,}] (10% tolerance)"
    )

    # Required interface with CORRECT values
    parts.append(
        "## Required Output Structure\n"
        "```python\n"
        "import torch\n"
        "import torch.nn as nn\n\n"
        "class YourModel(nn.Module):\n"
        f"    def __init__(self, context_len, prediction_len, num_variates, quantiles):\n"
        "        super().__init__()\n"
        "        # ... your architecture ...\n"
        "    def forward(self, x):\n"
        f"        # x: (batch, context_len, num_variates) = (batch, {CONTEXT_LEN}, {NUM_VARIATES})\n"
        f"        # return: (batch, prediction_len, num_variates, len(quantiles)) = "
        f"(batch, {PREDICTION_LEN}, {NUM_VARIATES}, {N_QUANTILES})\n"
        "        ...\n\n"
        "def build_model(context_len, prediction_len, num_variates, quantiles):\n"
        "    return YourModel(context_len, prediction_len, num_variates, quantiles)\n\n"
        "def build_optimizer(model):\n"
        "    return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)\n"
        "```\n\n"
        f"IMPORTANT: num_variates={NUM_VARIATES}, len(quantiles)={N_QUANTILES}.\n"
        f"quantiles = {QUANTILES}"
    )

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

    # Strategy
    strategy = BUCKET_STRATEGIES.get(bucket, "")
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
