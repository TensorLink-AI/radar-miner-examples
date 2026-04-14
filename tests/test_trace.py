"""Tests for the memory-efficient forward-pass tracer.

Covers:
  - Leaf-module entries are recorded with correct shape + param info.
  - Containers (Sequential, ModuleList) don't produce redundant entries.
  - No tensors are retained in the returned trace (ints/strings only).
  - max_entries cap prevents runaway memory on deep models.
  - format_trace renders a compact table.
  - Integration with the trace_architecture tool handler.
"""

import os
import sys
import tempfile
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "agents", "autonomous"))

for _k in list(sys.modules.keys()):
    if _k == "core" or _k.startswith("core."):
        del sys.modules[_k]
    if _k == "tools" or _k.startswith("tools."):
        del sys.modules[_k]

from core.trace import trace_architecture, format_trace
from tools import build_handlers


_TP = {
    "context_len": 64,
    "prediction_len": 16,
    "num_variates": 1,
    "quantiles": [0.1, 0.5, 0.9],
}

_CHALLENGE = {
    "task": {
        "task_params": dict(_TP),
        "constraints": [
            "Output: (batch, prediction_len, num_variates, len(quantiles))"
        ],
    },
}


# A simple model with a few distinct leaf ops so the trace is predictable.
_MODEL_CODE = '''\
import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self, context_len, prediction_len, num_variates, n_q):
        super().__init__()
        self.pred = prediction_len
        self.nv = num_variates
        self.nq = n_q
        self.net = nn.Sequential(
            nn.Linear(context_len, 32),
            nn.ReLU(),
            nn.Linear(32, prediction_len * n_q),
        )
        self.norm = nn.LayerNorm(prediction_len * n_q)

    def forward(self, x):
        b, L, V = x.shape
        h = self.net(x.transpose(1, 2))
        h = self.norm(h)
        return h.view(b, V, self.pred, self.nq).permute(0, 2, 1, 3)

def build_model(context_len, prediction_len, num_variates, quantiles):
    return M(context_len, prediction_len, num_variates, len(quantiles))

def build_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)
'''


# ── Core trace behaviour ─────────────────────────────────────────────

class TestTraceArchitecture:
    def test_returns_leaf_entries(self):
        entries, err = trace_architecture(_MODEL_CODE, _CHALLENGE)
        assert err == ""
        ops = [e["op"] for e in entries]
        # Leaf modules: 2x Linear, 1x ReLU, 1x LayerNorm. Sequential itself
        # should NOT appear because it's a container.
        assert ops.count("Linear") == 2
        assert "ReLU" in ops
        assert "LayerNorm" in ops
        assert "Sequential" not in ops

    def test_shapes_are_recorded(self):
        entries, err = trace_architecture(_MODEL_CODE, _CHALLENGE)
        assert err == ""
        first_linear = next(e for e in entries if e["op"] == "Linear")
        # Input to first Linear: (B=1, V=1, context_len=64)
        assert first_linear["input_shape"] == (1, 1, 64)
        assert first_linear["output_shape"] == (1, 1, 32)

    def test_param_counts_positive_for_parametric_ops(self):
        entries, err = trace_architecture(_MODEL_CODE, _CHALLENGE)
        assert err == ""
        linears = [e for e in entries if e["op"] == "Linear"]
        assert all(e["params"] > 0 for e in linears)
        relus = [e for e in entries if e["op"] == "ReLU"]
        assert all(e["params"] == 0 for e in relus)

    def test_no_tensors_retained(self):
        """Memory-efficiency guarantee: entries contain only ints/strings/tuples."""
        import torch
        entries, err = trace_architecture(_MODEL_CODE, _CHALLENGE)
        assert err == ""
        for e in entries:
            for key in ("input_shape", "output_shape"):
                shape = e[key]
                assert shape is None or isinstance(shape, tuple)
                if shape is not None:
                    assert all(isinstance(d, int) for d in shape)
            assert isinstance(e["name"], str)
            assert isinstance(e["op"], str)
            assert isinstance(e["params"], int)
            # Sanity: no torch.Tensor snuck in anywhere.
            for v in e.values():
                assert not isinstance(v, torch.Tensor)

    def test_max_entries_cap(self):
        entries, err = trace_architecture(_MODEL_CODE, _CHALLENGE, max_entries=1)
        assert err == ""
        assert len(entries) == 1

    def test_invalid_code_returns_error(self):
        entries, err = trace_architecture("x = 1", _CHALLENGE)
        assert entries == []
        assert "build_model" in err

    def test_build_model_raises_is_reported(self):
        code = '''
def build_model(context_len, prediction_len, num_variates, quantiles):
    raise ValueError("boom")
def build_optimizer(model):
    return None
'''
        entries, err = trace_architecture(code, _CHALLENGE)
        assert entries == []
        assert "boom" in err

    def test_forward_pass_error_is_reported(self):
        """If forward() raises, we get a trace-level error, not a crash."""
        code = '''
import torch.nn as nn
class Bad(nn.Module):
    def __init__(self, context_len, prediction_len, num_variates, n_q):
        super().__init__()
        self.fc = nn.Linear(context_len, 8)
    def forward(self, x):
        return self.fc(x[:, :, :5])  # wrong input dim on purpose
def build_model(context_len, prediction_len, num_variates, quantiles):
    return Bad(context_len, prediction_len, num_variates, len(quantiles))
def build_optimizer(model):
    import torch
    return torch.optim.Adam(model.parameters(), lr=1e-3)
'''
        entries, err = trace_architecture(code, _CHALLENGE)
        assert "Forward pass failed" in err or entries == []


# ── Formatting ───────────────────────────────────────────────────────

class TestFormatTrace:
    def test_formats_table(self):
        entries, err = trace_architecture(_MODEL_CODE, _CHALLENGE)
        assert err == ""
        rendered = format_trace(entries)
        assert "Idx" in rendered
        assert "Linear" in rendered
        assert "Total leaf params" in rendered

    def test_empty_entries(self):
        assert "no trace entries" in format_trace([]).lower()

    def test_max_rows_truncates_display(self):
        entries, err = trace_architecture(_MODEL_CODE, _CHALLENGE)
        assert err == ""
        rendered = format_trace(entries, max_rows=1)
        assert "more entries omitted" in rendered
        # Total still reflects all ops, not just the displayed one.
        total_from_full = sum(int(e.get("params") or 0) for e in entries)
        assert f"{total_from_full:,}" in rendered


# ── Tool-handler integration ─────────────────────────────────────────

class _NoopClient:
    def get_json(self, url): return {}
    def post_json(self, url, payload): return {}
    def get(self, url): return b""
    def put(self, url, data, content_type=None): return None


class TestToolHandler:
    def _handlers(self):
        challenge = {
            "min_flops_equivalent": 100_000,
            "max_flops_equivalent": 500_000,
            "feasible_frontier": [],
            "task": {
                "name": "ts_forecasting",
                "task_params": dict(_TP),
                "constraints": [
                    "Output: (batch, prediction_len, num_variates, len(quantiles))"
                ],
                "time_budget": 300,
            },
            "db_url": "",
            "desearch_url": "",
            "llm_url": "",
        }
        return build_handlers(
            _NoopClient(), challenge, tempfile.mkdtemp(), time.time() + 300,
        )

    def test_handler_registered(self):
        handlers = self._handlers()
        assert "trace_architecture" in handlers

    def test_handler_returns_table(self):
        handlers = self._handlers()
        result = handlers["trace_architecture"](code=_MODEL_CODE)
        assert "Linear" in result
        assert "Total leaf params" in result

    def test_handler_reports_bad_code(self):
        handlers = self._handlers()
        result = handlers["trace_architecture"](code="x = 1")
        assert "failed" in result.lower()
