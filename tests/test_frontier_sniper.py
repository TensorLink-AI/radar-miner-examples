"""Tests for frontier_sniper agent strategy logic."""

import importlib.util
import sys
import os
from unittest.mock import patch

# Load the agent module with a unique name to avoid cache collisions
_agent_dir = os.path.join(os.path.dirname(__file__), "..", "agents", "frontier_sniper")
sys.path.insert(0, _agent_dir)
_spec = importlib.util.spec_from_file_location(
    "frontier_sniper_agent", os.path.join(_agent_dir, "agent.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

get_frontier_for_bucket = _mod.get_frontier_for_bucket
analyze_frontier = _mod.analyze_frontier
build_strategy_instructions = _mod.build_strategy_instructions
update_playbook = _mod.update_playbook


class TestGetFrontierForBucket:
    def test_feasible_frontier(self):
        challenge = {"feasible_frontier": [{"code": "x=1"}]}
        result = get_frontier_for_bucket(challenge)
        assert len(result) == 1

    def test_pareto_frontier_fallback(self):
        challenge = {"pareto_frontier": [{"code": "x=2"}]}
        result = get_frontier_for_bucket(challenge)
        assert len(result) == 1

    def test_empty(self):
        assert get_frontier_for_bucket({}) == []

    def test_non_list(self):
        challenge = {"feasible_frontier": "invalid"}
        result = get_frontier_for_bucket(challenge)
        assert result == []


class TestAnalyzeFrontier:
    def test_empty_frontier(self):
        assert analyze_frontier([]) == ""

    def test_includes_metrics(self):
        frontier = [{
            "objectives": {"crps": 0.42, "mase": 0.55, "exec_time": 200, "memory_mb": 4000, "flops_equivalent_size": 1_000_000},
            "code": "model = Linear(10, 10)",
        }]
        result = analyze_frontier(frontier)
        assert "0.42" in result
        assert "model = Linear" in result
        assert "surgical" in result.lower() or "improvement" in result.lower()

    def test_truncates_long_code(self):
        frontier = [{"objectives": {}, "code": "x = 1\n" * 3000}]
        result = analyze_frontier(frontier)
        assert "truncated" in result


class TestBuildStrategyInstructions:
    def test_with_frontier(self):
        frontier = [{"objectives": {"crps": 0.4}, "code": "x=1"}]
        result = build_strategy_instructions(frontier, {}, "small")
        assert "sniping" in result.lower()

    def test_without_frontier(self):
        result = build_strategy_instructions([], {}, "small")
        assert "baseline" in result.lower()

    def test_includes_playbook(self):
        state = {"playbooks": {"small": "- cosine LR worked well"}}
        result = build_strategy_instructions([], state, "small")
        assert "cosine LR" in result

    def test_includes_history(self):
        state = {"history": [
            {"name": "exp1", "bucket": "small", "flops_target": 1000,
             "strategy": "sniper", "motivation": "test"}
        ]}
        result = build_strategy_instructions([], state, "small")
        assert "exp1" in result


class TestUpdatePlaybook:
    def test_creates_playbook(self):
        state = update_playbook({}, "small", "exp1", "cosine LR")
        assert "cosine LR" in state["playbooks"]["small"]

    def test_appends_to_playbook(self):
        state = {"playbooks": {"small": "- first entry"}}
        state = update_playbook(state, "small", "exp2", "warmup")
        assert "first entry" in state["playbooks"]["small"]
        assert "warmup" in state["playbooks"]["small"]

    def test_keeps_max_entries(self):
        state = {"playbooks": {"small": "\n".join(f"- entry{i}" for i in range(15))}}
        state = update_playbook(state, "small", "new", "latest")
        lines = [l for l in state["playbooks"]["small"].split("\n") if l.strip()]
        assert len(lines) <= 10


class TestIntegration:
    def test_extract_and_validate(self):
        """Integration test: code extraction + validation pipeline."""
        from core.llm import extract_code
        from core.validation import validate

        challenge = {
            "task": {"run_command": "python harness.py"},
            "flops_budget": {"min": 500_000, "max": 2_000_000},
        }

        raw = '''```python
import torch
import torch.nn as nn

class _Model(nn.Module):
    def __init__(self, context_len, prediction_len, num_variates, n_quantiles):
        super().__init__()
        self.prediction_len = prediction_len
        self.n_quantiles = n_quantiles
        # Small hidden dim to stay within the 500K-2M FLOPs bucket
        self.proj = nn.Linear(context_len, 3)
        self.out = nn.Linear(3, prediction_len * n_quantiles)

    def forward(self, x):
        # x: (batch, context_len, num_variates)
        h = self.proj(x.transpose(1, 2))  # (batch, num_variates, 3)
        out = self.out(h)                 # (batch, num_variates, pred*q)
        b, v, _ = out.shape
        out = out.view(b, v, self.prediction_len, self.n_quantiles)
        return out.permute(0, 2, 1, 3)

def build_model(context_len, prediction_len, num_variates, quantiles):
    return _Model(context_len, prediction_len, num_variates, len(quantiles))

def build_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)
```'''
        code = extract_code(raw)
        ok, errors = validate(code, challenge)
        assert ok, f"Validation failed: {errors}"
