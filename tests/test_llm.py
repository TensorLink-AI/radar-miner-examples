"""Tests for core.llm module — extract_code and chat with mock client."""

import pytest
import sys
import os
from unittest.mock import MagicMock

# Use any agent dir — all contain identical core/ modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agents", "frontier_sniper"))

from core.llm import extract_code, chat, get_models


class TestExtractCode:
    def test_python_fence(self):
        text = "Here is code:\n```python\nx = 42\n```\nDone."
        assert extract_code(text) == "x = 42"

    def test_py_fence(self):
        text = "```py\ny = 99\n```"
        assert extract_code(text) == "y = 99"

    def test_bare_fence(self):
        text = "```\nz = 0\n```"
        assert extract_code(text) == "z = 0"

    def test_no_fence(self):
        text = "x = 42"
        assert extract_code(text) == "x = 42"

    def test_multiline_code(self):
        text = "```python\ndef foo():\n    return 1\n```"
        assert "def foo():" in extract_code(text)
        assert "return 1" in extract_code(text)

    def test_first_block_extracted(self):
        text = "```python\nfirst = 1\n```\ntext\n```python\nsecond = 2\n```"
        assert extract_code(text) == "first = 1"

    def test_strips_whitespace(self):
        text = "```python\n\n  x = 1  \n\n```"
        result = extract_code(text)
        assert "x = 1" in result

    def test_Python_capitalized(self):
        text = "```Python\nx = 1\n```"
        assert extract_code(text) == "x = 1"


class TestChat:
    def test_calls_client_post_json(self):
        client = MagicMock()
        client.post_json.return_value = {"content": "Hello!", "remaining_queries": 10}
        result = chat(client, "http://llm:8080", [{"role": "user", "content": "hi"}])
        assert result == "Hello!"
        client.post_json.assert_called_once()
        call_url, call_payload = client.post_json.call_args[0]
        assert call_url == "http://llm:8080/chat"
        assert call_payload["messages"] == [{"role": "user", "content": "hi"}]

    def test_retries_on_failure(self):
        client = MagicMock()
        client.post_json.side_effect = [
            Exception("timeout"),
            {"content": "recovered", "remaining_queries": 5},
        ]
        result = chat(client, "http://llm:8080", [{"role": "user", "content": "hi"}])
        assert result == "recovered"
        assert client.post_json.call_count == 2

    def test_raises_after_max_retries(self):
        client = MagicMock()
        client.post_json.side_effect = Exception("always fails")
        with pytest.raises(RuntimeError, match="failed after"):
            chat(client, "http://llm:8080", [{"role": "user", "content": "hi"}])

    def test_no_llm_url(self):
        client = MagicMock()
        with pytest.raises(RuntimeError, match="No llm_url"):
            chat(client, "", [{"role": "user", "content": "hi"}])


class TestGetModels:
    def test_returns_model_list(self):
        client = MagicMock()
        client.get_json.return_value = ["model-a", "model-b"]
        result = get_models(client, "http://llm:8080")
        assert result == ["model-a", "model-b"]
        client.get_json.assert_called_once_with("http://llm:8080/models")

    def test_returns_models_from_dict(self):
        client = MagicMock()
        client.get_json.return_value = {"models": ["model-a"]}
        result = get_models(client, "http://llm:8080")
        assert result == ["model-a"]

    def test_graceful_failure(self):
        client = MagicMock()
        client.get_json.side_effect = Exception("network error")
        result = get_models(client, "http://llm:8080")
        assert result == []
