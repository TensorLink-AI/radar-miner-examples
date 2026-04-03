"""Tests for core.scratchpad module."""

import io
import json
import tarfile
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.scratchpad import load, save, MAX_SIZE


def make_tar_gz(state: dict) -> bytes:
    """Create a tar.gz archive containing state.json."""
    buf = io.BytesIO()
    state_bytes = json.dumps(state).encode()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="state.json")
        info.size = len(state_bytes)
        tar.addfile(info, io.BytesIO(state_bytes))
    return buf.getvalue()


class TestLoad:
    def test_no_url(self):
        assert load({}) == {}
        assert load({"scratchpad": {}}) == {}

    @patch("core.scratchpad.urllib.request.urlopen")
    def test_loads_state(self, mock_open):
        state = {"history": [{"name": "exp1"}]}
        data = make_tar_gz(state)

        mock_resp = MagicMock()
        mock_resp.read.return_value = data
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = mock_resp

        challenge = {"scratchpad": {"download_url": "https://r2.example.com/dl"}}
        result = load(challenge)
        assert result == state

    @patch("core.scratchpad.urllib.request.urlopen")
    def test_graceful_failure(self, mock_open):
        mock_open.side_effect = Exception("network error")
        result = load({"scratchpad": {"download_url": "https://r2.example.com/dl"}})
        assert result == {}


class TestSave:
    def test_no_url(self):
        assert save({}, {"key": "value"}) is False

    @patch("core.scratchpad.urllib.request.urlopen")
    def test_saves_state(self, mock_open):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = mock_resp

        challenge = {"scratchpad": {"upload_url": "https://r2.example.com/ul"}}
        result = save(challenge, {"key": "value"})
        assert result is True

    def test_max_size(self):
        assert MAX_SIZE == 10 * 1024 * 1024
