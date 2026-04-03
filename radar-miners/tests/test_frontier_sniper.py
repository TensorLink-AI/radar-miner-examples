"""Tests for frontier_sniper agent strategy logic."""

import json
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agents", "frontier_sniper"))

from agents.frontier_sniper.run import (
    get_frontier_for_bucket, analyze_frontier,
    build_strategy_instructions, update_playbook,
)


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
        assert result == []  # non-list returns empty


class TestAnalyzeFrontier:
    def test_empty_frontier(self):
        assert analyze_frontier([]) == ""

    def test_includes_metrics(self):
        frontier = [{
            "metrics": {"crps": 0.42, "mase": 0.55, "exec_time": 200, "memory_mb": 4000},
            "flops_equivalent_size": 1_000_000,
            "code": "model = Linear(10, 10)",
        }]
        result = analyze_frontier(frontier)
        assert "0.42" in result
        assert "model = Linear" in result
        assert "surgical" in result.lower() or "improvement" in result.lower()

    def test_truncates_long_code(self):
        frontier = [{"metrics": {}, "code": "x = 1\n" * 3000}]
        result = analyze_frontier(frontier)
        assert "truncated" in result


class TestBuildStrategyInstructions:
    def test_with_frontier(self):
        frontier = [{"metrics": {"crps": 0.4}, "code": "x=1"}]
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
    @patch("core.llm.chat")
    @patch("core.scratchpad.load")
    @patch("core.scratchpad.save")
    def test_main_with_mocked_llm(self, mock_save, mock_load, mock_chat):
        """Integration test: mock LLM returns valid harness code."""
        mock_load.return_value = {}
        mock_save.return_value = True
        mock_chat.return_value = '''```python
import torch
import torch.nn as nn

def build_model(context_len, prediction_len, num_variates, quantiles):
    return nn.Linear(context_len * num_variates, prediction_len * len(quantiles))

def build_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)
```'''
        challenge = {
            "task": {"run_command": "python harness.py"},
            "flops_budget": {"min": 500_000, "max": 2_000_000},
            "feasible_frontier": [],
            "db_url": "",
            "scratchpad": {},
        }

        # Test that the code extraction + validation pipeline works
        from core.llm import extract_code
        from core.validation import validate

        code = extract_code(mock_chat.return_value)
        ok, errors = validate(code, challenge)
        assert ok, f"Validation failed: {errors}"
