"""Tests for core.db_client module — tests use mocked HTTP."""

import json
import sys
import os
from unittest.mock import patch, MagicMock
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.db_client import (
    recent_experiments, pareto_front, recent_failures,
    family_summaries, component_stats, dead_ends,
    search_experiments, similar_experiments,
)

BASE = "http://test-db:8080"


def mock_urlopen(response_data):
    """Create a mock urlopen context manager returning given data."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(response_data).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestRecentExperiments:
    @patch("core.db_client.urllib.request.urlopen")
    def test_returns_data(self, mock_open):
        data = [{"name": "exp1", "metrics": {"crps": 0.4}}]
        mock_open.return_value = mock_urlopen(data)
        result = recent_experiments(BASE, n=5)
        assert result == data

    @patch("core.db_client.urllib.request.urlopen")
    def test_graceful_failure(self, mock_open):
        mock_open.side_effect = Exception("connection refused")
        result = recent_experiments(BASE)
        assert result == {}


class TestParetoFront:
    @patch("core.db_client.urllib.request.urlopen")
    def test_returns_data(self, mock_open):
        data = {"members": [{"crps": 0.3}]}
        mock_open.return_value = mock_urlopen(data)
        result = pareto_front(BASE)
        assert result == data


class TestRecentFailures:
    @patch("core.db_client.urllib.request.urlopen")
    def test_returns_failures(self, mock_open):
        data = [{"name": "bad", "reason": "OOM"}]
        mock_open.return_value = mock_urlopen(data)
        result = recent_failures(BASE)
        assert result == data


class TestComponentStats:
    @patch("core.db_client.urllib.request.urlopen")
    def test_returns_stats(self, mock_open):
        data = {"components": [{"name": "conv1d", "success_rate": 0.8}]}
        mock_open.return_value = mock_urlopen(data)
        result = component_stats(BASE)
        assert result == data


class TestDeadEnds:
    @patch("core.db_client.urllib.request.urlopen")
    def test_returns_dead_ends(self, mock_open):
        data = {"dead_ends": [{"pattern": "deep_rnn", "reason": "vanishing grad"}]}
        mock_open.return_value = mock_urlopen(data)
        result = dead_ends(BASE)
        assert result == data


class TestSearchExperiments:
    @patch("core.db_client.urllib.request.urlopen")
    def test_search(self, mock_open):
        data = [{"name": "transformer_v2"}]
        mock_open.return_value = mock_urlopen(data)
        result = search_experiments(BASE, "transformer")
        assert result == data


class TestSimilarExperiments:
    @patch("core.db_client.urllib.request.urlopen")
    def test_similar(self, mock_open):
        data = [{"name": "similar_exp"}]
        mock_open.return_value = mock_urlopen(data)
        result = similar_experiments(BASE, "abc123")
        assert result == data
