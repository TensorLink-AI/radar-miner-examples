"""System / user prompt builders for the OpenAI-SDK agent.

Wraps the existing ``core.prompt_builder`` helpers so the OpenAI agent
gets the same sizing/shape guidance the autonomous agent uses without
duplicating that logic. The system prompt explains the task, FLOPs
budget, code-shape rules, and the tool-calling contract; the user
prompt kicks off the round.
"""
from __future__ import annotations

from core.history import extract_flops_budget, identify_bucket
from core.prompt_builder import _compute_sizing_guidance, _format_task_params


def build_system_prompt(challenge: dict, bucket: str | None = None) -> str:
    """System message that teaches the LLM how to use the agent's tools.

    Pulls task params, constraints, FLOPs budget, and architecture-agnostic
    sizing guidance straight out of the challenge dict.
    """
    task = challenge.get("task", {}) or {}
    tp = task.get("task_params", {}) or {}
    param_str = ", ".join(tp.keys()) if tp else "**task_params"
    task_name = task.get("name", "unknown")

    flops_min, flops_max = extract_flops_budget(challenge)
    bucket = bucket or identify_bucket(flops_min, flops_max)
    target = int(flops_max * 0.6) if flops_max else 0
    gate_min = int(flops_min * 0.9) if flops_min else 0
    gate_max = int(flops_max * 1.1) if flops_max else 0

    parts: list[str] = []

    parts.append(
        "You are an autonomous ML architecture designer competing on the "
        "Radar subnet. You have a small set of tools — research the task, "
        "validate candidate code, and submit. The loop ends when you call "
        "`submit` with validated code, or when the time budget runs out.\n\n"
        "Workflow:\n"
        "  1. Call `analyze_task` to read the task spec.\n"
        "  2. Call `list_frontier` (and `get_frontier_member`) to see what "
        "you're competing against.\n"
        "  3. Draft model code. Call `validate_code` on it. If it fails, "
        "iterate.\n"
        "  4. Call `submit` with the final code, a short name, and a "
        "motivation."
    )

    parts.append(
        f"## Task: {task_name}\n"
        + (task.get("description", "") + "\n\n" if task.get("description") else "")
        + f"Task parameters: {_format_task_params(tp)}\n"
        f"build_model signature: `def build_model({param_str})`"
    )

    domain = task.get("domain_system_prompt", "")
    if domain:
        parts.append(f"## Domain Context\n{domain}")

    constraints = task.get("constraints", []) or []
    if constraints:
        parts.append(
            "## Constraints\n" + "\n".join(f"- {c}" for c in constraints)
        )

    objectives = task.get("objectives", []) or []
    if objectives:
        parts.append(
            "## Objectives\n" + "\n".join(f"- {o}" for o in objectives)
        )

    parts.append(
        f"## FLOPs Budget\n"
        f"Bucket: {bucket}\n"
        f"Range: [{flops_min:,}, {flops_max:,}]\n"
        f"Target: ~{target:,} FLOPs (60% of max)\n"
        f"Hard gate: [{gate_min:,}, {gate_max:,}] (instant rejection outside)"
    )

    parts.append(_compute_sizing_guidance(challenge))

    parts.append(
        "## Code Requirements\n"
        f"1. `def build_model({param_str})` — top-level, returns nn.Module\n"
        "2. `def build_optimizer(model)` — top-level, returns Optimizer\n"
        "3. Only torch + stdlib — no external dependencies\n"
        "4. Read all dimensions from the build_model arguments — never "
        "hardcode shapes\n"
        "5. Always call `validate_code` before `submit`"
    )

    return "\n\n".join(parts)


def build_user_prompt(challenge: dict, bucket: str | None = None) -> str:
    """Initial user message that starts the round."""
    task = challenge.get("task", {}) or {}
    task_name = task.get("name", "unknown")
    flops_min, flops_max = extract_flops_budget(challenge)
    bucket = bucket or identify_bucket(flops_min, flops_max)
    frontier = (
        challenge.get("feasible_frontier")
        or challenge.get("pareto_frontier")
        or []
    )

    return (
        f"New round started. Task: {task_name}, bucket: {bucket}, "
        f"FLOPs range: [{flops_min:,}, {flops_max:,}]. "
        f"Frontier has {len(frontier)} model(s) to beat.\n\n"
        "Use your tools to research the task, then propose a model. "
        "Validate before you submit."
    )
