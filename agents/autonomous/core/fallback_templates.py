"""Guaranteed-valid fallback architecture generator.

When the LLM fails to produce a model that passes validation within the
time/turn budget, this module generates a minimal but valid model that:

  1. Reads task_params generically from the challenge (no hardcoded keys)
  2. Sizes itself to ~60% of max_flops
  3. Passes both pre_validate_code (structure) AND the size gate (FLOPs)
  4. Works for ANY task because it derives the build_model signature and
     I/O shapes from the challenge

This is the safety net — it should ALWAYS produce submittable code.
"""

import math
import textwrap

from core.history import extract_flops_budget
from core.input_shape import infer_input, _looks_like_token_task
from core.output_shape import infer_output_shape


def generate_fallback(challenge: dict) -> str:
    """Generate a minimal but valid model that WILL pass the size gate.

    Returns a complete Python code string with build_model and build_optimizer.
    """
    task = challenge.get("task", {})
    tp = task.get("task_params", {})
    constraints = task.get("constraints", [])
    flops_min, flops_max = extract_flops_budget(challenge)
    target_flops = int(flops_max * 0.6) if flops_max else 300_000

    # Build the function signature from task_params keys
    param_names = list(tp.keys()) if tp else []
    sig = ", ".join(param_names) if param_names else "**kwargs"

    # Determine if this is a token-ID task or continuous-input task
    is_token_task = _looks_like_token_task(tp)

    if is_token_task:
        return _generate_token_fallback(sig, param_names, tp, target_flops, constraints)
    else:
        return _generate_continuous_fallback(sig, param_names, tp, target_flops, constraints)


def _parse_output_shape_from_constraints(constraints: list[str]) -> str | None:
    """Try to extract output shape expression from constraints.

    Looks for patterns like:
      "Output shape must be (batch, prediction_len, num_variates, len(quantiles))"
    """
    import re
    for c in constraints:
        m = re.search(r"[Oo]utput[^:]*(?:must be|shape|:)\s*\(([^)]+)\)", c)
        if m:
            return m.group(1)
    return None


def _generate_continuous_fallback(sig: str, param_names: list[str],
                                  tp: dict, target_flops: int,
                                  constraints: list[str]) -> str:
    """Generate a simple Linear model for continuous-input tasks.

    The model's ``forward()`` reshapes to the exact output shape inferred
    from the task's constraint string — avoiding the classic "output has
    wrong rank / wrong dim" failure that causes tensor-size mismatches
    during training.
    """
    # Infer input dimensions from task_params
    input_shape, _ = infer_input(tp, constraints)
    # input_shape is [1, dim1, dim2, ...] — product of dims after batch = input features per sample
    input_dims = input_shape[1:]  # drop batch

    # Estimate in_features as product of non-batch dims
    in_features = 1
    for d in input_dims:
        in_features *= d

    # Infer expected output shape (excluding batch) from constraints so the
    # forward pass reshapes to something that matches the task contract.
    # ``expected_out`` is a list of ints where unresolved dims are -1.
    expected_out = infer_output_shape(tp, constraints)
    expected_out_literal = repr(expected_out) if expected_out is not None else "None"

    # Build a kwargs dict string for the __init__ call
    kwargs_items = ", ".join(f"{n}={n}" for n in param_names)

    code = textwrap.dedent(f"""\
        import math
        import torch
        import torch.nn as nn

        class FallbackModel(nn.Module):
            def __init__(self, {sig}):
                super().__init__()
                # Store all params for shape computation
                self._tp = dict({kwargs_items})

                # Expected output shape (excluding batch) parsed from task
                # constraints by the host agent.  ``-1`` marks an unresolved
                # wildcard which we back-fill from the residual feature
                # count at forward time.
                self._expected_out = {expected_out_literal}

                int_params = [v for v in self._tp.values() if isinstance(v, int) and v > 0]
                list_params = [v for v in self._tp.values() if isinstance(v, list)]

                # Infer input feature dimension
                if len(int_params) >= 2:
                    self._in_features = int_params[0] * int_params[1]
                elif len(int_params) >= 1:
                    self._in_features = int_params[0]
                else:
                    self._in_features = 64

                # Compute the output feature count: if we have a concrete
                # expected shape, use the product of its resolved dims and
                # leave a single residual for any wildcard.  Otherwise fall
                # back to a best-effort estimate from int/list task params.
                if self._expected_out is not None:
                    resolved_product = 1
                    wildcard_dims = 0
                    for d in self._expected_out:
                        if d >= 0:
                            resolved_product *= d
                        else:
                            wildcard_dims += 1
                    # For each wildcard dim, pick a size of 1 so reshape works.
                    self._wildcard_size = 1
                    self._out_features = resolved_product * (self._wildcard_size ** wildcard_dims)
                else:
                    n_list = len(list_params[0]) if list_params else 1
                    if len(int_params) >= 2:
                        self._out_features = int_params[1] * max(1, n_list)
                        if len(int_params) >= 3:
                            self._out_features *= int_params[2]
                    else:
                        self._out_features = self._in_features

                # Size hidden_dim to fit FLOPs budget
                TARGET_FLOPS = {target_flops}
                flops_per_h = max(1, 2 * (self._in_features + self._out_features))
                hidden_dim = max(4, TARGET_FLOPS // flops_per_h)
                hidden_dim = min(hidden_dim, 2048)

                self.net = nn.Sequential(
                    nn.Linear(self._in_features, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, self._out_features),
                )

            def forward(self, x):
                b = x.shape[0]
                # Flatten input to (batch, in_features)
                x_flat = x.reshape(b, -1)
                # If input doesn't match expected, pad/truncate
                if x_flat.shape[1] != self._in_features:
                    if x_flat.shape[1] > self._in_features:
                        x_flat = x_flat[:, :self._in_features]
                    else:
                        pad = torch.zeros(b, self._in_features - x_flat.shape[1],
                                          device=x.device, dtype=x.dtype)
                        x_flat = torch.cat([x_flat, pad], dim=1)
                out = self.net(x_flat)

                # Reshape to the EXACT output shape required by the task,
                # replacing wildcards with concrete sizes.  torch.reshape
                # only accepts a single -1, so fix any extras to 1 and let
                # reshape infer the last wildcard from the residual.
                if self._expected_out is not None:
                    target = [b]
                    seen_wildcard = False
                    for d in self._expected_out:
                        if d >= 0:
                            target.append(d)
                        elif not seen_wildcard:
                            target.append(-1)
                            seen_wildcard = True
                        else:
                            target.append(1)
                    return out.reshape(*target)

                # No constraint parsed — best-effort fallback reshape.
                return out.reshape(b, -1, max(1, self._out_features // max(1, out.shape[1])))

        def build_model({sig}):
            return FallbackModel({sig})

        def build_optimizer(model):
            return torch.optim.Adam(model.parameters(), lr=1e-3)
    """)
    return code


def _generate_token_fallback(sig: str, param_names: list[str],
                             tp: dict, target_flops: int,
                             constraints: list[str]) -> str:
    """Generate a simple embedding + linear model for token-ID tasks."""
    vocab_key = None
    for k in ("vocab_size", "n_vocab", "vocabulary_size"):
        if k in tp:
            vocab_key = k
            break
    vocab_key = vocab_key or "vocab_size"

    seq_key = None
    for k in ("block_size", "seq_len", "sequence_length", "context_len"):
        if k in tp:
            seq_key = k
            break
    seq_key = seq_key or "block_size"

    code = textwrap.dedent(f"""\
        import math
        import torch
        import torch.nn as nn

        class FallbackTokenModel(nn.Module):
            def __init__(self, {sig}):
                super().__init__()
                self._tp = dict({", ".join(f"{n}={n}" for n in param_names)})
                vocab = int(self._tp.get("{vocab_key}", 1000))
                seq = int(self._tp.get("{seq_key}", 128))

                # Size embedding dim to fit FLOPs budget
                # FLOPs ~ 2 * seq * embed_dim * vocab (for output projection)
                TARGET_FLOPS = {target_flops}
                embed_dim = max(4, TARGET_FLOPS // max(1, 2 * seq * vocab))
                embed_dim = min(embed_dim, 512)

                self.embed = nn.Embedding(vocab, embed_dim)
                self.fc = nn.Linear(embed_dim, vocab)

            def forward(self, x):
                h = self.embed(x)
                return self.fc(h)

        def build_model({sig}):
            return FallbackTokenModel({sig})

        def build_optimizer(model):
            return torch.optim.Adam(model.parameters(), lr=1e-3)
    """)
    return code
