"""Per-subagent system prompt builders — PLACEHOLDER.

The real prompts are ported from ``openai_sdk/prompts.py`` and
restructured for the subagent split at the final checkpoint. For now
each builder returns a short stub so the orchestrator skeleton can
compile and round-trip end-to-end against a mocked LLM.
"""
from __future__ import annotations

from core.history import extract_flops_budget, identify_bucket


def build_researcher_system_prompt(
    challenge: dict, bucket: str | None = None,
) -> str:
    """STUB. Final wording lands after the prompt-discussion checkpoint."""
    flops_min, flops_max = extract_flops_budget(challenge)
    bucket = bucket or identify_bucket(flops_min, flops_max)
    task_name = (challenge.get("task", {}) or {}).get("name", "unknown")
    return (
        "You are the researcher subagent. Investigate the task and "
        "produce a JSON brief with relevant_prior_work, frontier_gaps, "
        "ideas_to_try, and a 3-5 step plan.\n\n"
        f"Task: {task_name}, bucket: {bucket}, "
        f"FLOPs range: [{flops_min:,}, {flops_max:,}]."
    )


def build_researcher_user_prompt(challenge: dict) -> str:
    return (
        "Use analyze_task / list_frontier / query_db / search_papers "
        "as you see fit. When you're ready, return ONLY a single "
        "fenced ```json block with the brief — no commentary."
    )


def build_designer_system_prompt(
    challenge: dict, bucket: str | None = None,
) -> str:
    """STUB."""
    flops_min, flops_max = extract_flops_budget(challenge)
    bucket = bucket or identify_bucket(flops_min, flops_max)
    return (
        "You are the designer subagent. Generate a build_model + "
        "build_optimizer that lands inside the FLOPs gate, validate "
        "it, and submit.\n\n"
        f"Bucket: {bucket}, FLOPs: [{flops_min:,}, {flops_max:,}]."
    )


def build_designer_user_prompt(challenge: dict, brief: dict) -> str:
    """STUB. The brief is rendered as JSON and dropped into the user
    prompt so the designer sees the researcher's plan."""
    import json as _json
    return (
        "Researcher brief:\n```json\n"
        + _json.dumps(brief, indent=2)
        + "\n```\n\nFollow the plan, but use your judgment — sketch, "
        "size, validate, submit."
    )


def build_critic_system_prompt() -> str:
    """STUB."""
    return (
        "You are the critic subagent. You see the designer's current "
        "code and the latest validate_code result. Your job is to "
        "give the designer one short, structured critique it can act "
        "on immediately.\n\n"
        "Reply with EXACTLY three lines:\n"
        "KEEP: <one sentence — what is working / should not be touched>\n"
        "CHANGE: <one sentence — the single most impactful revision>\n"
        "DROP: <one sentence — what to remove or stop doing>\n\n"
        "If validation passed (`ok ...`), KEEP the architecture and "
        "tell the designer to submit. If validation failed, focus "
        "CHANGE on the specific failing constraint."
    )


def build_critic_prompt(code: str, validation_result: str) -> str:
    """STUB."""
    return (
        f"validation result:\n{validation_result}\n\n"
        f"current code:\n```python\n{code[:4000]}\n```"
    )
