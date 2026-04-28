"""Critic subagent — STUB.

A single text-completion call (no tools) between designer iterations.
Input: the most recent ``validate_code`` result + the current code.
Output: structured ``keep / change / drop`` feedback the orchestrator
injects into the designer's next user-role turn.

This stub returns an empty critique — fleshed out at the next checkpoint.
"""
from __future__ import annotations


def run_critic(
    *,
    code: str,
    validation_result: str,
    deadline: float,
    llm_kwargs: dict,
) -> str:
    """Return a short critique string. Stub returns "" so the
    orchestrator can be wired end-to-end before the LLM call lands."""
    _ = (code, validation_result, deadline, llm_kwargs)
    return ""
