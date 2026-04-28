"""Designer subagent — STUB.

The designer takes the researcher's brief, generates code, validates
it, and submits. Tools: ``sketch_architecture``, ``estimate_layer_flops``,
``validate_code``, ``submit``.

Critic hand-offs are stitched in by the orchestrator between turns.
The submit-blocking hook lives in ``hooks.py`` and is wired in at
construction time.

This stub returns ``None`` — wiring lands at the next checkpoint.
"""
from __future__ import annotations

from typing import Optional

try:
    from ..tools import SubmitSignal
except ImportError:
    from tools import SubmitSignal


def run_designer(
    *,
    challenge: dict,
    handlers: dict,
    deadline: float,
    llm_kwargs: dict,
    brief: dict,
    state: dict,
    bucket: str,
) -> Optional[SubmitSignal]:
    """Run the designer subagent. Returns the submit signal on success
    or ``None`` if the designer never shipped.

    STUB: returns ``None``. Wires up next checkpoint.
    """
    _ = (challenge, handlers, deadline, llm_kwargs, brief, state, bucket)
    return None
