"""Tests for core.prompt_builder module."""

import pytest
import sys
import os

# Use any agent dir — all contain identical core/ modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agents", "frontier_sniper"))

# Clear cached core modules to ensure correct resolution
for _k in list(sys.modules.keys()):
    if _k == "core" or _k.startswith("core."):
        del sys.modules[_k]

from core.prompt_builder import (
    build_system_prompt, build_user_prompt, format_frontier, format_db_context
)

SAMPLE_CHALLENGE = {
    "task": {
        "run_command": "python harness.py",
        "domain_system_prompt": "You are building a time-series forecaster.",
        "constraints": ["Must use PyTorch", "No external data"],
        "objectives": ["Minimize CRPS", "Minimize MASE"],
        "anti_patterns": ["Don't use RNNs for long sequences"],
        "example_hypotheses": ["Patch embeddings reduce sequence length"],
        "task_params": {
            "context_len": 512, "prediction_len": 96,
            "num_variates": 1,
            "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        },
    },
    "flops_budget": {"min": 500_000, "max": 2_000_000},
}


FLAT_CHALLENGE = {
    "task": {
        "run_command": "python harness.py",
        "domain_system_prompt": "Time-series forecaster.",
        "task_params": {
            "context_len": 512, "prediction_len": 96,
            "num_variates": 1,
            "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        },
    },
    "min_flops_equivalent": 10_000_000,
    "max_flops_equivalent": 50_000_000,
}


class TestBuildSystemPrompt:
    def test_includes_domain(self):
        prompt = build_system_prompt(SAMPLE_CHALLENGE)
        assert "time-series forecaster" in prompt

    def test_includes_constraints(self):
        prompt = build_system_prompt(SAMPLE_CHALLENGE)
        assert "Must use PyTorch" in prompt

    def test_includes_objectives(self):
        prompt = build_system_prompt(SAMPLE_CHALLENGE)
        assert "Minimize CRPS" in prompt

    def test_includes_anti_patterns(self):
        prompt = build_system_prompt(SAMPLE_CHALLENGE)
        assert "Don't use RNNs" in prompt

    def test_includes_strategy_preamble(self):
        prompt = build_system_prompt(SAMPLE_CHALLENGE, "SNIPER MODE")
        assert "SNIPER MODE" in prompt

    def test_includes_absolute_requirements(self):
        prompt = build_system_prompt(SAMPLE_CHALLENGE)
        assert "build_model" in prompt
        assert "build_optimizer" in prompt
        assert "REJECTED" in prompt

    def test_empty_challenge(self):
        prompt = build_system_prompt({})
        assert isinstance(prompt, str)
        assert "build_model" in prompt


class TestBuildUserPrompt:
    def test_includes_flops_budget(self):
        prompt = build_user_prompt(SAMPLE_CHALLENGE)
        assert "500,000" in prompt
        assert "2,000,000" in prompt

    def test_includes_target_60pct(self):
        prompt = build_user_prompt(SAMPLE_CHALLENGE)
        assert "1,200,000" in prompt  # 60% of 2M

    def test_harness_interface(self):
        prompt = build_user_prompt(SAMPLE_CHALLENGE)
        assert "build_model" in prompt
        assert "build_optimizer" in prompt

    def test_includes_frontier_context(self):
        prompt = build_user_prompt(SAMPLE_CHALLENGE, frontier_context="FRONTIER DATA")
        assert "FRONTIER DATA" in prompt

    def test_includes_strategy(self):
        prompt = build_user_prompt(SAMPLE_CHALLENGE, strategy_instructions="DO X")
        assert "DO X" in prompt

    def test_output_format_instruction(self):
        prompt = build_user_prompt(SAMPLE_CHALLENGE)
        assert "```python" in prompt

    def test_flat_challenge_flops(self):
        prompt = build_user_prompt(FLAT_CHALLENGE)
        assert "10,000,000" in prompt
        assert "50,000,000" in prompt

    def test_includes_skeleton_example(self):
        prompt = build_user_prompt(SAMPLE_CHALLENGE)
        assert "def build_model(context_len, prediction_len, num_variates, quantiles):" in prompt
        assert "def build_optimizer(model):" in prompt
        assert "class MyModel(nn.Module):" in prompt

    def test_harness_explicit_shapes(self):
        prompt = build_user_prompt(SAMPLE_CHALLENGE)
        assert "(batch, 512, 1)" in prompt or "512" in prompt
        assert "build_model" in prompt
        assert "build_optimizer" in prompt
        assert "REJECTED" in prompt or "CRITICAL" in prompt

    def test_task_params_used_for_sizing(self):
        """Prompt should use task_params for sizing guidance, not defaults."""
        prompt = build_user_prompt(SAMPLE_CHALLENGE)
        # With num_variates=1 and 9 quantiles, max_hidden should be much larger
        # than with num_variates=370 and 3 quantiles
        assert "num_variates=1" in prompt
        assert "len(quantiles)=9" in prompt

    def test_includes_self_sizing_pattern(self):
        """Prompt should include the self-sizing pattern with TARGET_FLOPS."""
        prompt = build_user_prompt(SAMPLE_CHALLENGE)
        assert "TARGET_FLOPS" in prompt
        assert "flops_per_hidden" in prompt
        assert "hidden_dim" in prompt

    def test_different_task_params_different_sizing(self):
        """Two challenges with different task_params should produce different sizing."""
        ch1 = {
            "task": {"task_params": {"context_len": 512, "prediction_len": 96,
                     "num_variates": 1, "quantiles": [0.1, 0.5, 0.9]}},
            "flops_budget": {"min": 500_000, "max": 2_000_000},
        }
        ch2 = {
            "task": {"task_params": {"context_len": 512, "prediction_len": 96,
                     "num_variates": 370, "quantiles": [0.1, 0.5, 0.9]}},
            "flops_budget": {"min": 500_000, "max": 2_000_000},
        }
        prompt1 = build_user_prompt(ch1)
        prompt2 = build_user_prompt(ch2)
        assert "num_variates=1" in prompt1
        assert "num_variates=370" in prompt2
        # Different num_variates should produce different max_hidden values
        assert prompt1 != prompt2


class TestFormatFrontier:
    def test_empty_frontier(self):
        result = format_frontier([])
        assert "bootstrapping" in result.lower()

    def test_single_member(self):
        frontier = [{
            "objectives": {"crps": 0.42, "mase": 0.55, "exec_time": 200, "memory_mb": 4000, "flops_equivalent_size": 1_000_000},
            "code": "x = 1",
        }]
        result = format_frontier(frontier)
        assert "0.42" in result
        assert "x = 1" in result

    def test_truncates_long_code(self):
        frontier = [{
            "objectives": {"crps": 0.5},
            "code": "x = 1\n" * 2000,
        }]
        result = format_frontier(frontier)
        assert "truncated" in result

    def test_max_entries(self):
        frontier = [{"objectives": {"crps": i}, "code": ""} for i in range(10)]
        result = format_frontier(frontier, max_entries=2)
        assert "Member 1" in result
        assert "Member 2" in result
        assert "Member 3" not in result


class TestFormatDbContext:
    def test_empty_data(self):
        result = format_db_context({}, {}, {}, {})
        assert "No DB context" in result

    def test_recent_experiments_list(self):
        recent = [{"name": "exp1", "objectives": {"crps": 0.4}, "flops": 1000}]
        result = format_db_context(recent, {}, {}, {})
        assert "exp1" in result

    def test_recent_experiments_dict(self):
        recent = {"experiments": [{"name": "exp1", "objectives": {"crps": 0.4}, "flops": 1000}]}
        result = format_db_context(recent, {}, {}, {})
        assert "exp1" in result

    def test_failures(self):
        failures = [{"name": "bad_model", "reason": "OOM"}]
        result = format_db_context({}, failures, {}, {})
        assert "bad_model" in result
        assert "OOM" in result
