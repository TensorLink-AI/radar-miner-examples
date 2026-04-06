"""Tests for bucket_specialist agent strategy logic."""

import importlib.util
import sys
import os

# Load the agent module with a unique name to avoid cache collisions
_agent_dir = os.path.join(os.path.dirname(__file__), "..", "agents", "bucket_specialist")
sys.path.insert(0, _agent_dir)
_spec = importlib.util.spec_from_file_location(
    "bucket_specialist_agent", os.path.join(_agent_dir, "agent.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

get_bucket_template_prompt = _mod.get_bucket_template_prompt
build_strategy_instructions = _mod.build_strategy_instructions
save_template = _mod.save_template
BUCKET_TEMPLATES = _mod.BUCKET_TEMPLATES


class TestBucketTemplates:
    def test_all_buckets_have_templates(self):
        for bucket in ["tiny", "small", "medium_small", "medium", "large"]:
            assert bucket in BUCKET_TEMPLATES
            assert "description" in BUCKET_TEMPLATES[bucket]
            assert "tips" in BUCKET_TEMPLATES[bucket]


class TestGetBucketTemplatePrompt:
    def test_static_template(self):
        result = get_bucket_template_prompt("tiny", {})
        assert "100K-500K" in result

    def test_saved_template(self):
        state = {
            "templates": {"tiny": "model = SmallMLP()"},
            "template_metrics": {"tiny": {"crps": 0.45}},
        }
        result = get_bucket_template_prompt("tiny", state)
        assert "SmallMLP" in result
        assert "0.45" in result

    def test_unknown_bucket(self):
        result = get_bucket_template_prompt("unknown", {})
        assert result == ""


class TestBuildStrategyInstructions:
    def test_with_frontier(self):
        frontier = [{"objectives": {"crps": 0.4}}]
        result = build_strategy_instructions(
            frontier, {}, "small", 500_000, 2_000_000
        )
        assert "SPECIALIST" in result
        assert "small" in result

    def test_without_frontier(self):
        result = build_strategy_instructions(
            [], {}, "small", 500_000, 2_000_000
        )
        assert "No frontier" in result

    def test_target_flops(self):
        result = build_strategy_instructions(
            [], {}, "small", 500_000, 2_000_000
        )
        assert "1,200,000" in result  # 60% of 2M

    def test_includes_bucket_history(self):
        state = {"history": [
            {"name": "exp1", "bucket": "small", "flops_target": 1000,
             "strategy": "specialist", "motivation": "m"}
        ]}
        result = build_strategy_instructions(
            [], state, "small", 500_000, 2_000_000
        )
        assert "exp1" in result


class TestSaveTemplate:
    def test_saves_code(self):
        state = save_template({}, "tiny", "x = 1")
        assert state["templates"]["tiny"] == "x = 1"

    def test_saves_metrics(self):
        state = save_template({}, "tiny", "x = 1", {"crps": 0.4})
        assert state["template_metrics"]["tiny"]["crps"] == 0.4

    def test_overwrites(self):
        state = {"templates": {"tiny": "old"}}
        state = save_template(state, "tiny", "new")
        assert state["templates"]["tiny"] == "new"
