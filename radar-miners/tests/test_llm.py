"""Tests for core.llm module — extract_code only (chat requires network)."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.llm import extract_code


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
