"""Researcher subagent — STUB.

The researcher is the first subagent in the orchestrator's sequence.
It gets ``analyze_task``, ``list_frontier``, ``query_db``, and
``search_papers`` and produces a JSON brief the designer consumes:

    {
      "relevant_prior_work": [...],
      "frontier_gaps": [...],
      "ideas_to_try": [...],
      "plan": ["step 1", "step 2", "step 3"]
    }

This stub returns a hard-coded default brief so the orchestrator
skeleton compiles end-to-end. Real prompt + JSON parse + retry-once
landing in the next checkpoint.
"""
from __future__ import annotations

from typing import Optional


DEFAULT_BRIEF = {
    "relevant_prior_work": [],
    "frontier_gaps": [],
    "ideas_to_try": [],
    "plan": [
        "sketch a small candidate that reads dimensions from task_params",
        "size the candidate to land mid-bucket",
        "validate and submit",
    ],
}


def run_researcher(
    *,
    challenge: dict,
    handlers: dict,
    deadline: float,
    llm_kwargs: dict,
    state: dict,
    bucket: str,
) -> dict:
    """Return a researcher brief.

    STUB: returns ``DEFAULT_BRIEF``. Fleshed out at the next
    checkpoint with a real LLM call, JSON parse, and retry-once
    fallback.
    """
    _ = (challenge, handlers, deadline, llm_kwargs, state, bucket)
    return dict(DEFAULT_BRIEF)


def default_brief(challenge: dict, bucket: str) -> dict:
    """Construct a default brief from sizing guidance when the LLM
    fails to produce one. Used as the retry fallback."""
    _ = (challenge, bucket)
    return dict(DEFAULT_BRIEF)
