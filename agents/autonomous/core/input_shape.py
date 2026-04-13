"""Generic input shape inference from task_params and constraints.

Determines the correct input tensor shape and dtype for any task,
so the FLOPs estimator can run a forward pass without hardcoded
assumptions about specific tasks like ts_forecasting or nanogpt.
"""

import re

import torch


# ── Known task heuristics ────────────────────────────────────────
# These are soft pattern matchers, NOT hardcoded task names.  They
# look at the *keys* present in task_params to infer the role of
# each parameter.  New tasks with different key names will fall
# through to the generic path.

_TOKEN_ID_KEYS = frozenset({
    "vocab_size", "n_vocab", "vocabulary_size",
})

_SEQUENCE_LENGTH_KEYS = frozenset({
    "block_size", "seq_len", "sequence_length", "max_seq_len",
    "context_len", "context_length", "max_length",
})

_CHANNEL_KEYS = frozenset({
    "num_variates", "n_channels", "in_channels", "num_channels",
    "input_channels", "n_features", "num_features",
})


def _looks_like_token_task(tp: dict) -> bool:
    """Does this task take discrete token IDs as input?"""
    return bool(_TOKEN_ID_KEYS & set(tp.keys()))


def _find_key(tp: dict, candidates: frozenset) -> str | None:
    """Find the first key in tp that matches one of the candidates."""
    for k in tp:
        if k in candidates:
            return k
    return None


def _parse_constraints_for_shape(constraints: list[str]) -> list[int] | None:
    """Try to extract input shape from constraint strings.

    Looks for patterns like:
      "Model input: (batch, context_len, num_variates)"
      "Input shape: (B, block_size)"
      "Input tensor shape: (batch, seq_len)"
    """
    for c in constraints:
        m = re.search(
            r"[Ii]nput[^:]*:\s*\(([^)]+)\)", c
        )
        if m:
            dims = [d.strip() for d in m.group(1).split(",")]
            # Skip the batch dimension (first), return the rest as
            # hints (the caller resolves names to values from tp)
            return dims[1:] if len(dims) > 1 else dims
    return None


def infer_input(tp: dict, constraints: list[str] | None = None,
                ) -> tuple[list[int], torch.dtype]:
    """Infer model input shape and dtype from task_params.

    Returns (shape_including_batch, dtype).
    The batch dimension is always 1.

    Strategy:
      1. Token-ID tasks (has vocab_size etc.) -> (1, seq_len) LongTensor
      2. Continuous-input tasks with seq + channels -> (1, seq, channels)
      3. Constraint-string parsing for shape hints
      4. Generic fallback: use integer params as dimensions
    """
    if constraints is None:
        constraints = []

    # ── 1. Token-ID task (nanogpt-style) ─────────────────────────
    if _looks_like_token_task(tp):
        seq_key = _find_key(tp, _SEQUENCE_LENGTH_KEYS)
        seq_len = int(tp[seq_key]) if seq_key else 128
        return [1, seq_len], torch.long

    # ── 2. Sequence + channel task (ts_forecasting-style) ────────
    seq_key = _find_key(tp, _SEQUENCE_LENGTH_KEYS)
    ch_key = _find_key(tp, _CHANNEL_KEYS)
    if seq_key and ch_key:
        return [1, int(tp[seq_key]), int(tp[ch_key])], torch.float32

    # ── 3. Parse constraints for shape hints ─────────────────────
    hint_dims = _parse_constraints_for_shape(constraints)
    if hint_dims:
        resolved = []
        for d in hint_dims:
            # Try to resolve dimension name to a value from tp
            if d in tp and isinstance(tp[d], int):
                resolved.append(int(tp[d]))
            else:
                # Try to parse as literal int
                try:
                    resolved.append(int(d))
                except ValueError:
                    # Unknown dim name not in tp — skip constraint parsing
                    resolved = []
                    break
        if resolved:
            return [1] + resolved, torch.float32

    # ── 4. Single sequence-length key ────────────────────────────
    if seq_key:
        return [1, int(tp[seq_key])], torch.float32

    # ── 5. Generic fallback: use integer-valued params ───────────
    int_vals = [(k, v) for k, v in tp.items() if isinstance(v, int) and v > 0]
    if len(int_vals) >= 2:
        return [1, int_vals[0][1], int_vals[1][1]], torch.float32
    elif len(int_vals) == 1:
        return [1, int_vals[0][1]], torch.float32

    # Last resort — minimal 2D input
    return [1, 64], torch.float32
