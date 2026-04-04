"""Tests for core.db_client module — tests use mock GatedClient."""

import json
import sys
import os
from unittest.mock import MagicMock

# Use any agent dir — all contain identical core/ modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agents", "frontier_sniper"))

from core.db_client import (
    recent_experiments, pareto_front, recent_failures,
    family_summaries, component_stats, dead_ends,
    search_experiments, similar_experiments,
)

BASE = "http://test-db:8080"


def make_client(response_data):
    """Create a mock GatedClient returning given data for get_json/post_json."""
    client = MagicMock()
    client.get_json.return_value = response_data
    client.post_json.return_value = response_data
    return client


class TestRecentExperiments:
    def test_returns_data(self):
        data = [{"name": "exp1", "metrics": {"crps": 0.4}}]
        client = make_client(data)
        result = recent_experiments(client, BASE, n=5)
        assert result == data
        client.get_json.assert_called_once()

    def test_graceful_failure(self):
        client = MagicMock()
        client.get_json.side_effect = Exception("connection refused")
        result = recent_experiments(client, BASE)
        assert result == {}


class TestParetoFront:
    def test_returns_data(self):
        data = {"members": [{"crps": 0.3}]}
        client = make_client(data)
        result = pareto_front(client, BASE)
        assert result == data


class TestRecentFailures:
    def test_returns_failures(self):
        data = [{"name": "bad", "reason": "OOM"}]
        client = make_client(data)
        result = recent_failures(client, BASE)
        assert result == data


class TestComponentStats:
    def test_returns_stats(self):
        data = {"components": [{"name": "conv1d", "success_rate": 0.8}]}
        client = make_client(data)
        result = component_stats(client, BASE)
        assert result == data


class TestDeadEnds:
    def test_returns_dead_ends(self):
        data = {"dead_ends": [{"pattern": "deep_rnn", "reason": "vanishing grad"}]}
        client = make_client(data)
        result = dead_ends(client, BASE)
        assert result == data


class TestSearchExperiments:
    def test_search(self):
        data = [{"name": "transformer_v2"}]
        client = make_client(data)
        result = search_experiments(client, BASE, "transformer")
        assert result == data
        client.post_json.assert_called_once()


class TestSimilarExperiments:
    def test_similar(self):
        data = [{"name": "similar_exp"}]
        client = make_client(data)
        result = similar_experiments(client, BASE, "abc123")
        assert result == data
