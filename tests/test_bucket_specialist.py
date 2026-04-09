"""Tests for bucket_specialist agent strategy logic."""

import importlib.util
import sys
import os

# Load the agent module with a unique name to avoid cache collisions
_agent_dir = os.path.join(os.path.dirname(__file__), "..", "agents", "bucket_specialist")
sys.path.insert(0, _agent_dir)
# Clear cached core modules to ensure correct resolution
for _k in list(sys.modules.keys()):
    if _k == "core" or _k.startswith("core."):
        del sys.modules[_k]
_spec = importlib.util.spec_from_file_location(
    "bucket_specialist_agent", os.path.join(_agent_dir, "agent.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

get_bucket_template_prompt = _mod.get_bucket_template_prompt
build_strategy_instructions = _mod.build_strategy_instructions
save_template = _mod.save_template
_compute_bucket_guidance = _mod._compute_bucket_guidance


class TestBucketGuidance:
    def test_all_buckets_have_guidance(self):
        from core.history import SIZE_BUCKETS
        for bucket, (bmin, bmax) in SIZE_BUCKETS.items():
            challenge = {"task": {}, "flops_budget": {"min": bmin, "max": bmax}}
            guidance = _compute_bucket_guidance(bucket, bmin, bmax, challenge)
            assert "description" in guidance
            assert "tips" in guidance

    def test_guidance_contains_budget_info(self):
        challenge = {"task": {}, "flops_budget": {"min": 100_000, "max": 500_000}}
        guidance = _compute_bucket_guidance("tiny", 100_000, 500_000, challenge)
        assert "100,000" in guidance["description"]
        assert "500,000" in guidance["description"]


class TestGetBucketTemplatePrompt:
    def test_dynamic_template(self):
        challenge = {"task": {}, "flops_budget": {"min": 100_000, "max": 500_000}}
        result = get_bucket_template_prompt(
            "tiny", {}, flops_min=100_000, flops_max=500_000, challenge=challenge)
        assert "100,000" in result

    def test_saved_template(self):
        state = {
            "templates": {"tiny": "model = SmallMLP()"},
            "template_metrics": {"tiny": {"crps": 0.45}},
        }
        challenge = {"task": {}, "flops_budget": {"min": 100_000, "max": 500_000}}
        result = get_bucket_template_prompt(
            "tiny", state, flops_min=100_000, flops_max=500_000, challenge=challenge)
        assert "SmallMLP" in result
        assert "0.45" in result

    def test_unknown_bucket_no_challenge(self):
        result = get_bucket_template_prompt("unknown", {})
        assert result == ""


class TestBuildStrategyInstructions:
    def test_with_frontier(self):
        frontier = [{"objectives": {"crps": 0.4}}]
        challenge = {"task": {}, "flops_budget": {"min": 500_000, "max": 2_000_000}}
        result = build_strategy_instructions(
            frontier, {}, "small", 500_000, 2_000_000, challenge=challenge
        )
        assert "SPECIALIST" in result
        assert "small" in result

    def test_without_frontier(self):
        challenge = {"task": {}, "flops_budget": {"min": 500_000, "max": 2_000_000}}
        result = build_strategy_instructions(
            [], {}, "small", 500_000, 2_000_000, challenge=challenge
        )
        assert "No frontier" in result

    def test_target_flops(self):
        challenge = {"task": {}, "flops_budget": {"min": 500_000, "max": 2_000_000}}
        result = build_strategy_instructions(
            [], {}, "small", 500_000, 2_000_000, challenge=challenge
        )
        assert "1,200,000" in result  # 60% of 2M

    def test_includes_bucket_history(self):
        state = {"history": [
            {"name": "exp1", "bucket": "small", "flops_target": 1000,
             "strategy": "specialist", "motivation": "m"}
        ]}
        challenge = {"task": {}, "flops_budget": {"min": 500_000, "max": 2_000_000}}
        result = build_strategy_instructions(
            [], state, "small", 500_000, 2_000_000, challenge=challenge
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
