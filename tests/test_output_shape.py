"""Tests for output-shape inference and verification.

Exercises:
  - infer_output_shape parsing (3D, 4D, mixed, unparseable)
  - verify_output_shape rank and per-dim comparison (incl. wildcards)
  - Integration with core.validation.validate_code so bad-shape models are
    rejected before submission — regression guard for tensor-size mismatch
    training failures.
"""

import os
import sys

# Autonomous agent is the one we're extending.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "agents", "autonomous"))

# Clear cached core modules so we pick up autonomous agent's core.
for _k in list(sys.modules.keys()):
    if _k == "core" or _k.startswith("core."):
        del sys.modules[_k]

from core.output_shape import infer_output_shape, verify_output_shape
from core.validation import validate_code


_TS_TP = {
    "context_len": 512,
    "prediction_len": 96,
    "num_variates": 1,
    "quantiles": [0.1, 0.5, 0.9],
}

_TS_CONSTRAINTS = [
    "Output shape must be (batch, prediction_len, num_variates, len(quantiles))",
]


def _ts_challenge(bucket=None, constraints=None):
    ch = {
        "task": {
            "task_params": dict(_TS_TP),
            "constraints": constraints if constraints is not None else list(_TS_CONSTRAINTS),
        },
    }
    if bucket == "tiny":
        ch["min_flops_equivalent"] = 100_000
        ch["max_flops_equivalent"] = 500_000
    return ch


# ── Inference ───────────────────────────────────────────────────────

class TestInferOutputShape:
    def test_4d_ts_forecasting(self):
        shape = infer_output_shape(_TS_TP, _TS_CONSTRAINTS)
        assert shape == [96, 1, 3]

    def test_3d_token_task(self):
        tp = {"block_size": 128, "vocab_size": 1000}
        constraints = ["Output: (batch, block_size, vocab_size)"]
        shape = infer_output_shape(tp, constraints)
        assert shape == [128, 1000]

    def test_2d_classification(self):
        tp = {"num_classes": 10}
        constraints = ["Output shape: (batch, num_classes)"]
        shape = infer_output_shape(tp, constraints)
        assert shape == [10]

    def test_len_expression(self):
        tp = {"targets": [0, 1, 2, 3, 4]}
        constraints = ["Output: (batch, len(targets))"]
        shape = infer_output_shape(tp, constraints)
        assert shape == [5]

    def test_literal_int_dim(self):
        tp = {"n": 16}
        constraints = ["Output: (batch, n, 3)"]
        assert infer_output_shape(tp, constraints) == [16, 3]

    def test_unresolved_dim_becomes_wildcard(self):
        tp = {"prediction_len": 64}
        constraints = ["Output: (batch, prediction_len, mystery)"]
        assert infer_output_shape(tp, constraints) == [64, -1]

    def test_no_constraint_returns_none_for_unknown_task(self):
        # Unknown task params + no constraint + unknown name → can't infer.
        tp = {"foo": 10, "bar": 20}
        assert infer_output_shape(tp, []) is None

    def test_constraint_without_output_returns_none_for_unknown_task(self):
        tp = {"foo": 10, "bar": 20}
        assert infer_output_shape(tp, ["Use Adam", "No dropout"]) is None

    def test_none_constraints_returns_none_for_unknown_task(self):
        tp = {"foo": 10, "bar": 20}
        assert infer_output_shape(tp, None) is None

    # ── Fingerprint fallback (no constraint string) ────────────────

    def test_fingerprint_ts_forecasting_by_name(self):
        shape = infer_output_shape(_TS_TP, [], task_name="ts_forecasting")
        assert shape == [96, 1, 3]

    def test_fingerprint_ts_forecasting_by_keys_no_name(self):
        # Even without a recognisable task name, the key set alone triggers
        # the ts_forecasting fingerprint — the CLAUDE.md output shape is
        # fully determined by task_params.
        shape = infer_output_shape(_TS_TP, [], task_name="")
        assert shape == [96, 1, 3]

    def test_fingerprint_token_task(self):
        tp = {"block_size": 128, "vocab_size": 5000}
        shape = infer_output_shape(tp, [], task_name="nanogpt")
        assert shape == [128, 5000]

    def test_fingerprint_token_task_alt_key(self):
        tp = {"seq_len": 64, "n_vocab": 2000}
        shape = infer_output_shape(tp, [], task_name="")
        assert shape == [64, 2000]

    def test_fingerprint_unknown_task_returns_none(self):
        tp = {"foo": 10, "bar": 20}
        assert infer_output_shape(tp, [], task_name="mystery") is None

    def test_constraint_string_wins_over_fingerprint(self):
        # If both are parseable, the explicit constraint takes precedence.
        shape = infer_output_shape(
            _TS_TP,
            ["Output: (batch, 7, 1, 3)"],
            task_name="ts_forecasting",
        )
        assert shape == [7, 1, 3]


# ── Verification ────────────────────────────────────────────────────

class TestVerifyOutputShape:
    def test_matches(self):
        assert verify_output_shape((1, 96, 1, 3), [96, 1, 3]) is None

    def test_batch_dim_is_free(self):
        # Any batch size is fine.
        assert verify_output_shape((7, 96, 1, 3), [96, 1, 3]) is None

    def test_rank_mismatch_3d_vs_4d(self):
        err = verify_output_shape((1, 96, 3), [96, 1, 3])
        assert err is not None
        assert "rank mismatch" in err.lower()
        assert "4D" in err and "3D" in err

    def test_rank_mismatch_4d_vs_3d(self):
        err = verify_output_shape((1, 96, 1, 3, 2), [96, 1, 3])
        assert err is not None
        assert "rank mismatch" in err.lower()

    def test_dim_mismatch_surfaces_actual_vs_expected(self):
        # The regression case from the bug report: prediction_len=64 expected
        # but model produced 96.
        err = verify_output_shape((1, 96, 1, 3), [64, 1, 3])
        assert err is not None
        assert "expected 64" in err and "got 96" in err

    def test_wildcard_dim_skipped(self):
        assert verify_output_shape((1, 96, 42, 3), [96, -1, 3]) is None

    def test_none_actual(self):
        assert verify_output_shape(None, [1]) is not None


# ── Integration with validate_code ──────────────────────────────────

# A model that outputs the WRONG shape: (B, context_len, ...) instead of
# (B, prediction_len, ...) — the classic "tensor a (96) vs tensor b (64)"
# training failure if prediction_len != context_len.
_WRONG_SHAPE_CODE = '''\
import torch
import torch.nn as nn

class Wrong(nn.Module):
    def __init__(self, context_len, prediction_len, num_variates, n_q):
        super().__init__()
        self.context_len = context_len
        self.num_variates = num_variates
        self.n_q = n_q
        # Intentionally project to context_len, not prediction_len.
        self.fc = nn.Linear(context_len, context_len * n_q)

    def forward(self, x):
        b, L, V = x.shape
        h = self.fc(x.transpose(1, 2))  # (B, V, context_len * n_q)
        return h.view(b, V, self.context_len, self.n_q).permute(0, 2, 1, 3)

def build_model(context_len, prediction_len, num_variates, quantiles):
    return Wrong(context_len, prediction_len, num_variates, len(quantiles))

def build_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)
'''

# A correct model that outputs (B, prediction_len, num_variates, n_q).
_CORRECT_CODE = '''\
import torch
import torch.nn as nn

class Right(nn.Module):
    def __init__(self, context_len, prediction_len, num_variates, n_q):
        super().__init__()
        self.pred = prediction_len
        self.nv = num_variates
        self.nq = n_q
        self.fc = nn.Linear(context_len, prediction_len * n_q)

    def forward(self, x):
        b, L, V = x.shape
        h = self.fc(x.transpose(1, 2))
        return h.view(b, V, self.pred, self.nq).permute(0, 2, 1, 3)

def build_model(context_len, prediction_len, num_variates, quantiles):
    return Right(context_len, prediction_len, num_variates, len(quantiles))

def build_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)
'''


class TestValidateCodeShapeIntegration:
    def test_wrong_shape_is_rejected(self):
        ok, errors = validate_code(_WRONG_SHAPE_CODE, _ts_challenge())
        assert not ok
        joined = "\n".join(errors)
        assert ("mismatch" in joined.lower()) or ("rank" in joined.lower()), joined

    def test_correct_shape_passes(self):
        ok, errors = validate_code(_CORRECT_CODE, _ts_challenge())
        assert ok, f"Unexpected errors: {errors}"

    def test_fingerprint_catches_bug_when_no_constraint_string(self):
        # Regression: validator log showed "tensor a (96) must match tensor b (64)"
        # on a Medium-bucket round. The previous check silently skipped when
        # the challenge had no "Output: (...)" constraint string. With the
        # task-fingerprint fallback we now catch the wrong-shape model even
        # when constraints is empty, as long as the task is recognizable.
        ch = _ts_challenge(constraints=[])
        # The ts_forecasting task name + prediction_len/num_variates/quantiles
        # in task_params are enough to know the expected output shape.
        ch["task"]["name"] = "ts_forecasting"
        ok, errors = validate_code(_WRONG_SHAPE_CODE, ch)
        assert not ok, "Expected shape check to fire via fingerprint fallback"
        joined = "\n".join(errors)
        assert "mismatch" in joined.lower()

    def test_flops_and_shape_both_enforced(self):
        # With a tiny bucket, the FLOPs check should still run alongside
        # the shape check on the correct model.
        ok, errors = validate_code(_CORRECT_CODE, _ts_challenge(bucket="tiny"))
        # Model may be over the tiny bucket; but if it fails it must be a
        # FLOPs error, not a shape error.
        if not ok:
            joined = "\n".join(errors)
            assert "FLOPs" in joined or "flops" in joined.lower()
