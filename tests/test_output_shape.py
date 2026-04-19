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

    def test_no_constraint_returns_none(self):
        assert infer_output_shape(_TS_TP, []) is None

    def test_constraint_without_output_returns_none(self):
        assert infer_output_shape(_TS_TP, ["Use Adam", "No dropout"]) is None

    def test_none_constraints_returns_none(self):
        assert infer_output_shape(_TS_TP, None) is None


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

    def test_no_constraints_means_no_shape_check(self):
        # Remove the output-shape constraint — the check should be skipped.
        ch = _ts_challenge(constraints=[])
        ok, errors = validate_code(_WRONG_SHAPE_CODE, ch)
        # The wrong-shape model has no FLOPs problem, so without a shape
        # constraint to trigger verification it should pass structural checks.
        assert ok, f"Unexpected errors without constraints: {errors}"

    def test_flops_and_shape_both_enforced(self):
        # With a tiny bucket, the FLOPs check should still run alongside
        # the shape check on the correct model.
        ok, errors = validate_code(_CORRECT_CODE, _ts_challenge(bucket="tiny"))
        # Model may be over the tiny bucket; but if it fails it must be a
        # FLOPs error, not a shape error.
        if not ok:
            joined = "\n".join(errors)
            assert "FLOPs" in joined or "flops" in joined.lower()


# ══════════════════════════════════════════════════════════════════════
#  Regression guard: "tensor a (48) must match tensor b (96) at dim 1"
#
# Two miner rounds shipped models that passed the single-batch shape
# check but crashed during training with a broadcast mismatch between
# the model output and the task-declared target. This happened because
# the bug was either batch-size-dependent or only surfaced during the
# backward pass. The trainability check (forward+loss+backward at
# batch_size=2) must catch these classes.
# ══════════════════════════════════════════════════════════════════════

# Model whose forward collapses prediction_len to half when batch_size>1.
# At batch=1 the old validator saw (1, 96, 1, 3) — all good. At batch=2
# the output is (2, 48, 1, 3) and mse_loss against the task's target
# shape (2, 96, 1, 3) raises the exact 48-vs-96 error that failed in
# production.
_BATCH_DEP_HALFPRED_CODE = '''\
import torch
import torch.nn as nn

class BatchDepHalfPred(nn.Module):
    def __init__(self, context_len, prediction_len, num_variates, n_q):
        super().__init__()
        self.pred = prediction_len
        self.nv = num_variates
        self.nq = n_q
        self.fc = nn.Linear(context_len, prediction_len * n_q)

    def forward(self, x):
        b, L, V = x.shape
        h = self.fc(x.transpose(1, 2))  # (b, V, pred*nq)
        if b > 1:
            # Bug: at training batch sizes the model silently halves the
            # prediction horizon — the classic failure mode.
            half = self.pred // 2
            h = h[..., : half * self.nq]
            return h.view(b, V, half, self.nq).permute(0, 2, 1, 3)
        return h.view(b, V, self.pred, self.nq).permute(0, 2, 1, 3)

def build_model(context_len, prediction_len, num_variates, quantiles):
    return BatchDepHalfPred(context_len, prediction_len, num_variates, len(quantiles))

def build_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)
'''


# Model that hits a backward-only autograd failure. Forward pass and
# mse_loss both succeed, but the detached output breaks the backward
# graph — the kind of bug a no-grad forward would miss.
_BACKWARD_FAILURE_CODE = '''\
import torch
import torch.nn as nn

class BackwardBreaker(nn.Module):
    def __init__(self, context_len, prediction_len, num_variates, n_q):
        super().__init__()
        self.pred = prediction_len
        self.nv = num_variates
        self.nq = n_q
        self.fc = nn.Linear(context_len, prediction_len * n_q)

    def forward(self, x):
        b, L, V = x.shape
        h = self.fc(x.transpose(1, 2))
        h = h.view(b, V, self.pred, self.nq).permute(0, 2, 1, 3).contiguous()
        return h.detach()

def build_model(context_len, prediction_len, num_variates, quantiles):
    return BackwardBreaker(context_len, prediction_len, num_variates, len(quantiles))

def build_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)
'''


class TestTrainabilityCheckCatchesRuntimeFailures:
    """The trainability check (step 10) re-runs the model with grads
    enabled at batch_size=2 against a target of the task-declared shape.

    These tests encode the failure modes from the production training
    crashes — they must stay rejected so we never ship a model that
    burns training wall-clock only to crash with the same error.
    """

    def test_48_vs_96_broadcast_is_caught(self):
        ok, errors = validate_code(_BATCH_DEP_HALFPRED_CODE, _ts_challenge())
        assert not ok, "The 48 vs 96 broadcast bug must not pass validation"
        joined = "\n".join(errors).lower()
        # The error surfaces either as the batch=2 forward-pass shape
        # (48 vs 96) or the loss broadcast error — both are acceptable
        # signals, both prevent submission.
        assert (
            "48" in joined
            or "broadcast" in joined
            or "training-time" in joined
            or "loss computation failed" in joined
            or "mismatch" in joined
        ), f"Expected a training-time / shape-broadcast error, got: {errors}"

    def test_backward_only_failure_is_caught(self):
        ok, errors = validate_code(_BACKWARD_FAILURE_CODE, _ts_challenge())
        # We only require rejection here — the exact backward error is
        # torch-version-dependent. What matters is we reject pre-submit
        # instead of letting training crash.
        assert not ok, (
            "Backward-only failures must be caught at validate time so "
            "the agent can retry before shipping broken code"
        )

    def test_correct_model_still_passes_with_trainability_check(self):
        # The stronger check must not produce false positives on models
        # that are actually fine.
        ok, errors = validate_code(_CORRECT_CODE, _ts_challenge())
        assert ok, f"Correct model rejected by trainability check: {errors}"
