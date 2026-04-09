"""Tests for simple_modeler agent strategy logic."""

import importlib.util
import sys
import os
from unittest.mock import patch

# Load the agent module with a unique name to avoid cache collisions
_agent_dir = os.path.join(os.path.dirname(__file__), "..", "agents", "simple_modeler")
sys.path.insert(0, _agent_dir)
# Clear cached core modules to ensure correct resolution
for _k in list(sys.modules.keys()):
    if _k == "core" or _k.startswith("core."):
        del sys.modules[_k]
_spec = importlib.util.spec_from_file_location(
    "simple_modeler_agent", os.path.join(_agent_dir, "agent.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

get_frontier = _mod.get_frontier
build_strategy_instructions = _mod.build_strategy_instructions


class TestGetFrontier:
    def test_feasible_frontier(self):
        challenge = {"feasible_frontier": [{"code": "x=1"}]}
        result = get_frontier(challenge)
        assert len(result) == 1

    def test_pareto_frontier_fallback(self):
        challenge = {"pareto_frontier": [{"code": "x=2"}]}
        result = get_frontier(challenge)
        assert len(result) == 1

    def test_empty(self):
        assert get_frontier({}) == []

    def test_non_list(self):
        challenge = {"feasible_frontier": "invalid"}
        result = get_frontier(challenge)
        assert result == []


class TestBuildStrategyInstructions:
    def test_tiny_bucket_guidance(self):
        challenge = {"task": {}, "flops_budget": {"min": 100_000, "max": 500_000}}
        result = build_strategy_instructions([], {}, "tiny", 100_000, 500_000, challenge=challenge)
        assert "max hidden" in result.lower()
        assert "flops" in result.lower()

    def test_small_bucket_guidance(self):
        challenge = {"task": {}, "flops_budget": {"min": 500_000, "max": 2_000_000}}
        result = build_strategy_instructions([], {}, "small", 500_000, 2_000_000, challenge=challenge)
        assert "small" in result.lower()

    def test_medium_bucket_guidance(self):
        challenge = {"task": {}, "flops_budget": {"min": 10_000_000, "max": 50_000_000}}
        result = build_strategy_instructions([], {}, "medium", 10_000_000, 50_000_000, challenge=challenge)
        assert "max hidden" in result.lower()

    def test_large_bucket_guidance(self):
        challenge = {"task": {}, "flops_budget": {"min": 50_000_000, "max": 125_000_000}}
        result = build_strategy_instructions([], {}, "large", 50_000_000, 125_000_000, challenge=challenge)
        assert "max hidden" in result.lower()

    def test_with_frontier_shows_best_crps(self):
        frontier = [
            {"objectives": {"crps": 0.42}},
            {"objectives": {"crps": 0.50}},
        ]
        challenge = {"task": {}, "flops_budget": {"min": 500_000, "max": 2_000_000}}
        result = build_strategy_instructions(frontier, {}, "small", 500_000, 2_000_000, challenge=challenge)
        assert "0.42" in result

    def test_without_frontier_says_baseline(self):
        challenge = {"task": {}, "flops_budget": {"min": 500_000, "max": 2_000_000}}
        result = build_strategy_instructions([], {}, "small", 500_000, 2_000_000, challenge=challenge)
        assert "baseline" in result.lower()

    def test_includes_target_flops(self):
        challenge = {"task": {}, "flops_budget": {"min": 500_000, "max": 2_000_000}}
        result = build_strategy_instructions([], {}, "small", 500_000, 2_000_000, challenge=challenge)
        # Target is 60% of max = 1,200,000
        assert "1,200,000" in result

    def test_includes_bucket_history(self):
        state = {"history": [
            {"name": "exp1", "bucket": "small", "flops_target": 1000,
             "strategy": "simple", "motivation": "test"}
        ]}
        challenge = {"task": {}, "flops_budget": {"min": 500_000, "max": 2_000_000}}
        result = build_strategy_instructions([], state, "small", 500_000, 2_000_000, challenge=challenge)
        assert "exp1" in result

    def test_no_gaming_language(self):
        """Ensure strategy instructions don't include gaming/exploitation language."""
        frontier = [{"objectives": {"crps": 0.4}, "code": "x=1"}]
        challenge = {"task": {}, "flops_budget": {"min": 500_000, "max": 2_000_000}}
        result = build_strategy_instructions(frontier, {}, "small", 500_000, 2_000_000, challenge=challenge)
        result_lower = result.lower()
        assert "sniping" not in result_lower
        assert "dominate" not in result_lower
        assert "exploit" not in result_lower


class TestIntegration:
    def test_extract_and_validate(self):
        """Integration test: code extraction + validation pipeline."""
        from core.llm import extract_code
        from core.validation import validate

        challenge = {
            "task": {"run_command": "python harness.py", "task_params": {
                "context_len": 512, "prediction_len": 96,
                "num_variates": 1,
                "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            }},
            "flops_budget": {"min": 500_000, "max": 2_000_000},
        }

        raw = '''```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, context_len, prediction_len, num_variates, n_quantiles):
        super().__init__()
        self.prediction_len = prediction_len
        self.n_quantiles = n_quantiles
        # Hidden dim sized for V=1, nq=9, 500K-2M FLOPs bucket
        self.proj = nn.Linear(context_len, 300)
        self.out = nn.Linear(300, prediction_len * n_quantiles)

    def forward(self, x):
        # x: (batch, context_len, num_variates)
        h = self.proj(x.transpose(1, 2))  # (batch, num_variates, 300)
        out = self.out(h)                 # (batch, num_variates, pred*q)
        b, v, _ = out.shape
        out = out.view(b, v, self.prediction_len, self.n_quantiles)
        return out.permute(0, 2, 1, 3)

def build_model(context_len, prediction_len, num_variates, quantiles):
    return SimpleModel(context_len, prediction_len, num_variates, len(quantiles))

def build_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)
```'''
        code = extract_code(raw)
        ok, errors = validate(code, challenge)
        assert ok, f"Validation failed: {errors}"
