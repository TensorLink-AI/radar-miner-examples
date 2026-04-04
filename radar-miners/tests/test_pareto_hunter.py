"""Tests for pareto_hunter agent strategy logic."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agents", "pareto_hunter"))

from agents.pareto_hunter.agent import (
    analyze_frontier_weaknesses, get_dominatable_targets,
    build_strategy_instructions, OBJECTIVE_WEIGHTS,
)


class TestObjectiveWeights:
    def test_weights_defined(self):
        assert OBJECTIVE_WEIGHTS["crps"] == 1.0
        assert OBJECTIVE_WEIGHTS["mase"] == 0.5
        assert OBJECTIVE_WEIGHTS["exec_time"] == 0.2
        assert OBJECTIVE_WEIGHTS["memory_mb"] == 0.1


class TestAnalyzeFrontierWeaknesses:
    def test_empty_frontier(self):
        result = analyze_frontier_weaknesses([])
        assert "No frontier" in result

    def test_slow_member(self):
        frontier = [{"metrics": {"crps": 0.4, "exec_time": 250, "memory_mb": 1000}}]
        result = analyze_frontier_weaknesses(frontier)
        assert "SLOW" in result

    def test_memory_hog(self):
        frontier = [{"metrics": {"crps": 0.4, "exec_time": 50, "memory_mb": 5000}}]
        result = analyze_frontier_weaknesses(frontier)
        assert "MEMORY HOG" in result or "HEAVY" in result

    def test_weak_mase(self):
        frontier = [{"metrics": {"crps": 0.4, "mase": 0.8, "exec_time": 50, "memory_mb": 500}}]
        result = analyze_frontier_weaknesses(frontier)
        assert "MASE" in result

    def test_no_weaknesses(self):
        frontier = [{"metrics": {"crps": 0.3, "mase": 0.3, "exec_time": 30, "memory_mb": 500}}]
        result = analyze_frontier_weaknesses(frontier)
        assert "No obvious weaknesses" in result

    def test_includes_code(self):
        frontier = [{"metrics": {}, "code": "x = efficient_model()"}]
        result = analyze_frontier_weaknesses(frontier)
        assert "efficient_model" in result

    def test_truncates_long_code(self):
        frontier = [{"metrics": {}, "code": "x = 1\n" * 3000}]
        result = analyze_frontier_weaknesses(frontier)
        assert "truncated" in result


class TestGetDominatableTargets:
    def test_empty(self):
        assert get_dominatable_targets([]) == []

    def test_identifies_slow(self):
        frontier = [
            {"metrics": {"exec_time": 200, "memory_mb": 500, "mase": 0.3}},
            {"metrics": {"exec_time": 30, "memory_mb": 500, "mase": 0.3}},
        ]
        targets = get_dominatable_targets(frontier)
        assert len(targets) == 1
        assert targets[0]["metrics"]["exec_time"] == 200

    def test_identifies_memory_hog(self):
        frontier = [
            {"metrics": {"exec_time": 50, "memory_mb": 5000, "mase": 0.3}},
        ]
        targets = get_dominatable_targets(frontier)
        assert len(targets) == 1

    def test_sorted_by_weakness(self):
        frontier = [
            {"metrics": {"exec_time": 200, "memory_mb": 5000, "mase": 0.8}},  # 3 weaknesses
            {"metrics": {"exec_time": 200, "memory_mb": 500, "mase": 0.3}},   # 1 weakness
        ]
        targets = get_dominatable_targets(frontier)
        assert len(targets) == 2
        # Most weak first
        assert targets[0]["_weakness_score"] >= targets[1]["_weakness_score"]

    def test_no_weaknesses(self):
        frontier = [
            {"metrics": {"exec_time": 30, "memory_mb": 500, "mase": 0.3}},
        ]
        assert get_dominatable_targets(frontier) == []


class TestBuildStrategyInstructions:
    def test_with_dominatable_frontier(self):
        frontier = [{"metrics": {"crps": 0.4, "exec_time": 300, "memory_mb": 5000}}]
        result = build_strategy_instructions(frontier, {}, "small")
        assert "DOMINATABLE" in result

    def test_without_frontier(self):
        result = build_strategy_instructions([], {}, "small")
        assert "efficient" in result.lower() or "bfloat16" in result.lower()

    def test_includes_pareto_history(self):
        state = {"history": [
            {"name": "ph1", "bucket": "small", "flops_target": 1000,
             "strategy": "pareto_hunter", "motivation": "multi-obj"}
        ]}
        result = build_strategy_instructions([], state, "small")
        assert "ph1" in result

    def test_non_pareto_history_excluded(self):
        state = {"history": [
            {"name": "sniper1", "bucket": "small", "flops_target": 1000,
             "strategy": "frontier_sniper", "motivation": "test"}
        ]}
        result = build_strategy_instructions([], state, "small")
        assert "sniper1" not in result
