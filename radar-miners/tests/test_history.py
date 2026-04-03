"""Tests for core.history module."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.history import (
    get_history, add_entry, get_bucket_history, format_history,
    identify_bucket, SIZE_BUCKETS,
)


class TestGetHistory:
    def test_empty_state(self):
        assert get_history({}) == []

    def test_existing_history(self):
        state = {"history": [{"name": "test"}]}
        assert len(get_history(state)) == 1


class TestAddEntry:
    def test_adds_entry(self):
        state = {}
        state = add_entry(state, name="exp1", code="x=1",
                          motivation="test", bucket="small")
        assert len(state["history"]) == 1
        assert state["history"][0]["name"] == "exp1"

    def test_keeps_max_50(self):
        state = {"history": [{"name": f"exp{i}"} for i in range(55)]}
        state = add_entry(state, name="new", code="x=1",
                          motivation="test")
        assert len(state["history"]) == 50

    def test_entry_fields(self):
        state = add_entry({}, name="n", code="c", motivation="m",
                          bucket="tiny", flops=100, strategy="sniper")
        entry = state["history"][0]
        assert entry["bucket"] == "tiny"
        assert entry["flops_target"] == 100
        assert entry["strategy"] == "sniper"
        assert "timestamp" in entry
        assert "code_hash" in entry


class TestGetBucketHistory:
    def test_filters_by_bucket(self):
        state = {
            "history": [
                {"name": "a", "bucket": "tiny"},
                {"name": "b", "bucket": "small"},
                {"name": "c", "bucket": "tiny"},
            ]
        }
        result = get_bucket_history(state, "tiny")
        assert len(result) == 2
        assert all(e["bucket"] == "tiny" for e in result)


class TestFormatHistory:
    def test_empty(self):
        assert "No previous" in format_history([])

    def test_formats_entries(self):
        entries = [{"name": "exp1", "bucket": "small",
                    "flops_target": 1000, "strategy": "sniper",
                    "motivation": "test improvement"}]
        result = format_history(entries)
        assert "exp1" in result
        assert "sniper" in result

    def test_max_entries(self):
        entries = [{"name": f"exp{i}", "bucket": "s", "flops_target": 0,
                    "strategy": "s", "motivation": "m"} for i in range(20)]
        result = format_history(entries, max_entries=3)
        lines = [l for l in result.split("\n") if l.strip()]
        assert len(lines) == 3


class TestIdentifyBucket:
    def test_exact_match_tiny(self):
        assert identify_bucket(100_000, 500_000) == "tiny"

    def test_exact_match_small(self):
        assert identify_bucket(500_000, 2_000_000) == "small"

    def test_exact_match_medium(self):
        assert identify_bucket(10_000_000, 50_000_000) == "medium"

    def test_exact_match_large(self):
        assert identify_bucket(50_000_000, 125_000_000) == "large"

    def test_fuzzy_match(self):
        # Close to small bucket
        result = identify_bucket(600_000, 1_800_000)
        assert result == "small"

    def test_all_buckets_defined(self):
        assert len(SIZE_BUCKETS) == 5
