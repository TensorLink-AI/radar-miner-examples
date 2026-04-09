"""Tests for FLOPs estimation and validation integration."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agents", "frontier_sniper"))
# Clear cached core modules to ensure correct resolution
for _k in list(sys.modules.keys()):
    if _k == "core" or _k.startswith("core."):
        del sys.modules[_k]

from core.flops_estimator import estimate_flops
from core.validation import validate


# Standard task_params matching GIFT-Eval production
_DEFAULT_TASK_PARAMS = {
    "context_len": 512,
    "prediction_len": 96,
    "num_variates": 1,
    "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}

# Challenge with a "small" bucket (500K-2M FLOPs)
SMALL_BUCKET_CHALLENGE = {
    "task": {"run_command": "python harness.py", "task_params": _DEFAULT_TASK_PARAMS},
    "flops_budget": {"min": 500_000, "max": 2_000_000},
}

# Challenge with a "tiny" bucket (100K-500K FLOPs)
TINY_BUCKET_CHALLENGE = {
    "task": {"run_command": "python harness.py", "task_params": _DEFAULT_TASK_PARAMS},
    "flops_budget": {"min": 100_000, "max": 500_000},
}

# Challenge with flat FLOPs fields
FLAT_FLOPS_CHALLENGE = {
    "task": {"run_command": "python harness.py", "task_params": _DEFAULT_TASK_PARAMS},
    "min_flops_equivalent": 500_000,
    "max_flops_equivalent": 2_000_000,
}

# Challenge with no FLOPs budget (validation should skip FLOPs check)
NO_FLOPS_CHALLENGE = {
    "task": {"run_command": "python harness.py", "task_params": _DEFAULT_TASK_PARAMS},
}

# A model that fits within the small bucket (~1M FLOPs)
SMALL_MODEL_CODE = '''\
import torch
import torch.nn as nn

class SmallModel(nn.Module):
    def __init__(self, context_len, prediction_len, num_variates, n_quantiles):
        super().__init__()
        self.prediction_len = prediction_len
        self.n_quantiles = n_quantiles
        self.linear = nn.Linear(context_len, prediction_len * n_quantiles)

    def forward(self, x):
        out = self.linear(x.transpose(1, 2))
        b, v, _ = out.shape
        out = out.view(b, v, self.prediction_len, self.n_quantiles)
        return out.permute(0, 2, 1, 3)

def build_model(context_len, prediction_len, num_variates, quantiles):
    return SmallModel(context_len, prediction_len, num_variates, len(quantiles))

def build_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)
'''

# A model that's way too large for the tiny bucket
LARGE_MODEL_CODE = '''\
import torch
import torch.nn as nn

class LargeModel(nn.Module):
    def __init__(self, context_len, prediction_len, num_variates, n_quantiles):
        super().__init__()
        self.prediction_len = prediction_len
        self.n_quantiles = n_quantiles
        self.fc1 = nn.Linear(context_len, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, prediction_len * n_quantiles)

    def forward(self, x):
        out = self.fc1(x.transpose(1, 2))
        out = torch.relu(out)
        out = self.fc2(out)
        out = torch.relu(out)
        out = self.fc3(out)
        b, v, _ = out.shape
        out = out.view(b, v, self.prediction_len, self.n_quantiles)
        return out.permute(0, 2, 1, 3)

def build_model(context_len, prediction_len, num_variates, quantiles):
    return LargeModel(context_len, prediction_len, num_variates, len(quantiles))

def build_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)
'''


class TestEstimateFlops:
    def test_returns_positive_flops(self):
        flops, err = estimate_flops(SMALL_MODEL_CODE, SMALL_BUCKET_CHALLENGE)
        assert err == ""
        assert flops is not None
        assert flops > 0

    def test_invalid_code_returns_error(self):
        flops, err = estimate_flops("x = 1", NO_FLOPS_CHALLENGE)
        assert flops is None
        assert "build_model" in err

    def test_model_with_no_layers_returns_zero(self):
        code = '''\
import torch.nn as nn
class Empty(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x
def build_model(context_len, prediction_len, num_variates, quantiles):
    return Empty()
def build_optimizer(model):
    return None
'''
        flops, err = estimate_flops(code, NO_FLOPS_CHALLENGE)
        assert err == ""
        assert flops == 0

    def test_larger_model_has_more_flops(self):
        small_flops, _ = estimate_flops(SMALL_MODEL_CODE, NO_FLOPS_CHALLENGE)
        large_flops, _ = estimate_flops(LARGE_MODEL_CODE, NO_FLOPS_CHALLENGE)
        assert large_flops > small_flops


class TestValidateFlops:
    def test_no_flops_budget_skips_check(self):
        ok, errors = validate(SMALL_MODEL_CODE, NO_FLOPS_CHALLENGE)
        assert ok, f"Unexpected errors: {errors}"

    def test_model_too_large_for_bucket(self):
        ok, errors = validate(LARGE_MODEL_CODE, TINY_BUCKET_CHALLENGE)
        assert not ok
        assert any("above hard gate" in e for e in errors)

    def test_flat_flops_format_works(self):
        """Flat min_flops_equivalent/max_flops_equivalent fields are supported."""
        ok, errors = validate(LARGE_MODEL_CODE, FLAT_FLOPS_CHALLENGE)
        # Should either pass or fail with a FLOPs error (not crash)
        if not ok:
            assert any("FLOPs" in e or "flops" in e.lower() for e in errors)

    def test_existing_validation_still_works(self):
        """FLOPs check doesn't break structural validation."""
        ok, errors = validate("import subprocess", SMALL_BUCKET_CHALLENGE)
        assert not ok
        assert any("subprocess" in e for e in errors)
        # Should NOT also have FLOPs errors (structural errors skip FLOPs)
        assert not any("FLOPs" in e for e in errors)

    def test_estimator_reads_task_params(self):
        """Estimator uses task_params, not hardcoded defaults."""
        challenge = {
            "task": {"task_params": {"context_len": 512, "prediction_len": 96,
                     "num_variates": 1,
                     "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}},
            "min_flops_equivalent": 500_000,
            "max_flops_equivalent": 2_000_000,
        }
        flops, err = estimate_flops(SMALL_MODEL_CODE, challenge)
        assert err == ""
        assert flops is not None
        # With V=1, nq=9, this should be well under 2M (not 109M as with V=370)
        assert flops < 2_000_000, f"FLOPs {flops} should be < 2M with V=1, nq=9"

    def test_different_variates_different_flops(self):
        """Different num_variates in task_params produce different FLOPs."""
        ch1 = {
            "task": {"task_params": {"context_len": 512, "prediction_len": 96,
                     "num_variates": 1,
                     "quantiles": [0.1, 0.5, 0.9]}},
        }
        ch2 = {
            "task": {"task_params": {"context_len": 512, "prediction_len": 96,
                     "num_variates": 10,
                     "quantiles": [0.1, 0.5, 0.9]}},
        }
        flops1, _ = estimate_flops(SMALL_MODEL_CODE, ch1)
        flops2, _ = estimate_flops(SMALL_MODEL_CODE, ch2)
        # More variates means more FLOPs (model processes each variate)
        assert flops2 > flops1, f"V=10 ({flops2}) should have more FLOPs than V=1 ({flops1})"
