"""Tests for scratchpad state helpers in core.history module."""

import json
import os
import sys
import tempfile

# Use any agent dir — all contain identical core/ modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agents", "frontier_sniper"))
# Clear cached core modules to ensure correct resolution
for _k in list(sys.modules.keys()):
    if _k == "core" or _k.startswith("core."):
        del sys.modules[_k]

from core.history import load_state, save_state


class TestLoadState:
    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            assert load_state(d) == {}

    def test_loads_existing_state(self):
        with tempfile.TemporaryDirectory() as d:
            state = {"history": [{"name": "exp1"}]}
            with open(os.path.join(d, "state.json"), "w") as f:
                json.dump(state, f)
            assert load_state(d) == state


class TestSaveState:
    def test_saves_state(self):
        with tempfile.TemporaryDirectory() as d:
            state = {"key": "value", "count": 42}
            save_state(d, state)
            with open(os.path.join(d, "state.json")) as f:
                loaded = json.load(f)
            assert loaded == state

    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as d:
            nested = os.path.join(d, "sub", "dir")
            save_state(nested, {"x": 1})
            assert os.path.exists(os.path.join(nested, "state.json"))

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            state = {"history": [{"name": "exp1"}], "templates": {"tiny": "x=1"}}
            save_state(d, state)
            loaded = load_state(d)
            assert loaded == state
