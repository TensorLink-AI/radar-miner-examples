"""Task-aware prompt construction from challenge spec."""

import json
from core.history import extract_flops_budget


def _compute_sizing_guidance(challenge: dict) -> str:
    """Build architecture-agnostic FLOPs calculator section from the challenge.

    Teaches the LLM how to estimate FLOPs for common PyTorch primitives and
    provides budget-derived sizing constraints.  Does NOT prescribe any
    specific architecture.
    """
    task = challenge.get("task", {})
    tp = task.get("task_params", {})
    ctx_len = tp.get("context_len", 512)
    pred_len = tp.get("prediction_len", 96)
    n_var = tp.get("num_variates", 1)
    q_list = tp.get("quantiles", [])
    n_q = len(q_list)

    flops_min, flops_max = extract_flops_budget(challenge)
    target = int(flops_max * 0.6) if flops_max else 0
    gate_min = int(flops_min * 0.9) if flops_min else 0
    gate_max = int(flops_max * 1.1) if flops_max else 0

    # Compute max_hidden for a simple 2-layer channel-independent model
    # Linear(ctx_len, H) + Linear(H, pred*nq) per variate
    # FLOPs = V * 2 * (ctx_len * H + H * pred * nq) ≈ V * 2 * H * (ctx_len + pred*nq)
    denom = 2 * n_var * (ctx_len + pred_len * n_q)
    max_hidden = target // denom if denom > 0 else 0

    lines = [
        "### FLOPs Calculator (use this to self-check your design)",
        "",
        "**FLOPs formulas for common ops** (multiply-accumulate = 2 FLOPs per MAC):",
        f"- `nn.Linear(in, out)` on input `(batch, seq, in)` -> `2 * in * out * seq` FLOPs",
        f"- `nn.Conv1d(C_in, C_out, K)` on length L -> `2 * C_in * K * C_out * L_out` FLOPs",
        f"- `nn.MultiheadAttention(d, heads)` on seq S -> `8 * S * d^2 + 2 * S^2 * d` FLOPs",
        "- Note: if your model reshapes to `(batch * V, ...)`, the V factor is already in "
        "the batch -- do NOT double-count V in the per-layer formula",
        "",
        "**Budget summary:**",
        f"- Target FLOPs: {target:,} (60% of max)",
        f"- Hard gate: [{gate_min:,}, {gate_max:,}] (10% tolerance of [{flops_min:,}, {flops_max:,}])",
        f"- context_len={ctx_len}, num_variates={n_var}, prediction_len={pred_len}, "
        f"len(quantiles)={n_q}",
        "",
        "**Quick sizing reference** (computed from this budget):",
        f"- A simple 2-layer channel-independent model (Linear->ReLU->Linear) can afford "
        f"max hidden ~ {max_hidden}.",
        "- More complex architectures must budget FLOPs across their layers accordingly.",
        "- ALWAYS verify your total FLOPs against the gate range before finalizing.",
        "",
        "### Self-Sizing Pattern (IMPORTANT — use this in your build_model)",
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

    # Always reinforce the harness contract in the system prompt
    tp = challenge.get("task", {}).get("task_params", {})
    param_str = ", ".join(tp.keys()) if tp else "**task_params"
    parts.append(
        "### Absolute Requirements\n"
        "Every response MUST be a single ```python code block containing "
        f"top-level `def build_model({param_str})` "
        "and `def build_optimizer(model)` functions. Code missing either function is "
        "REJECTED automatically."
    )

    return "\n\n".join(parts)


def build_user_prompt(challenge: dict, *,
                      frontier_context: str = "",
                      db_context: str = "",
                      history_context: str = "",
                      strategy_instructions: str = "") -> str:
    """Build the user prompt with budget, frontier, DB context, and history."""
    parts: list[str] = []

    # FLOPs budget — supports both nested and flat challenge formats
    flops_min, flops_max = extract_flops_budget(challenge)
    target = int(flops_max * 0.6) if flops_max else 0
    parts.append(
        f"### FLOPs Budget\n"
        f"- Min: {flops_min:,}\n"
        f"- Max: {flops_max:,}\n"
        f"- Target ~60% of max: {target:,}\n"
        f"- Hard gate: [{int(flops_min * 0.9):,}, {int(flops_max * 1.1):,}]"
    )

    # Harness interface — use actual challenge parameters when available
    task = challenge.get("task", {})
    tp = task.get("task_params", {})
    ctx_len = tp.get("context_len", 512)
    pred_len = tp.get("prediction_len", 96)
    n_var = tp.get("num_variates", 1)
    q_list = tp.get("quantiles", [])
    n_q = len(q_list)
    param_str = ", ".join(tp.keys()) if tp else "**task_params"
    parts.append(
        "### Required Interface (Harness Task)\n"
        "Your code MUST define these two top-level functions (not inside a class):\n\n"
        f"1. `def build_model({param_str}):`\n"
        f"   - Called as: build_model({ctx_len}, {pred_len}, {n_var}, {q_list})\n"
        "   - Must return an nn.Module\n"
        f"   - Forward input shape:  (batch, {ctx_len}, {n_var})\n"
        f"   - Forward output shape: (batch, {pred_len}, {n_var}, {n_q})  "
        "# i.e. (batch, prediction_len, num_variates, len(quantiles))\n\n"
        "2. `def build_optimizer(model):`\n"
        "   - Takes the model returned by build_model()\n"
        "   - Must return a torch.optim.Optimizer\n\n"
        "Optional hooks: training_config(), init_weights(), configure_amp(), "
        "transform_batch(), on_step_end(), build_scheduler(), compute_loss()\n\n"
        "CRITICAL: If build_model or build_optimizer is missing, the code is "
        "REJECTED before evaluation. Both MUST be present as top-level def statements.\n\n"
        "Minimal skeleton (your code must follow this structure):\n"
        "```python\n"
        "import torch\n"
        "import torch.nn as nn\n\n"
        "class MyModel(nn.Module):\n"
        f"    def __init__(self, {param_str}):\n"
        "        super().__init__()\n"
        "        # ... your architecture ...\n"
        "    def forward(self, x):\n"
        "        # x: (batch, context_len, num_variates)\n"
        "        # return: (batch, prediction_len, num_variates, len(quantiles))\n"
        "        ...\n\n"
        f"def build_model({param_str}):\n"
        f"    return MyModel({param_str})\n\n"
        "def build_optimizer(model):\n"
        "    return torch.optim.Adam(model.parameters(), lr=1e-3)\n"
        "```"
    )

    # Dynamic FLOPs calculator section
    sizing = _compute_sizing_guidance(challenge)
    parts.append(sizing)

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
        metrics = member.get("objectives", {})
        lines.append(
            f"**Member {i + 1}**: "
            f"crps={metrics.get('crps', '?')}, "
            f"mase={metrics.get('mase', '?')}, "
            f"exec_time={metrics.get('exec_time', '?')}s, "
            f"memory={metrics.get('memory_mb', '?')}MB, "
            f"flops={member.get('objectives', {}).get('flops_equivalent_size', '?')}"
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
