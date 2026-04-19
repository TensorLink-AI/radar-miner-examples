"""Tests for the OpenAI-SDK reference agent.

The OpenAI client is mocked at module level — no real network calls.
The autonomous agent's `core/` modules sit on sys.path (set up by
``agents/openai_sdk/__init__.py``) so structural validation works
without per-test plumbing.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

# ── Module loading ────────────────────────────────────────────────
# Add the repo root so ``agents.openai_sdk`` resolves as a package.
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Other test modules add ``agents/autonomous/`` to sys.path under the
# name ``core``. We need that too — but importing the openai_sdk
# package handles it via its __init__.py. Clear cached ``core`` first
# in case a previous test bound a stale agent's core.
for _k in list(sys.modules.keys()):
    if _k == "core" or _k.startswith("core."):
        del sys.modules[_k]

# Drop any cached ``tools`` module from another agent's test run — the
# openai_sdk package has its own ``agents.openai_sdk.tools`` and that
# is what we want, not the ``tools`` autonomous test inserts.
for _k in list(sys.modules.keys()):
    if _k == "tools" or _k.startswith("tools."):
        del sys.modules[_k]


def _reset_llm_client_cache() -> None:
    """Reset the cached singleton between tests so each test sees a
    freshly constructed (and freshly mocked) OpenAI client."""
    from agents.openai_sdk import llm_client
    llm_client._cached_client = None
    llm_client._cached_config = None


def _reset_core_resolution() -> None:
    """Drop any cached ``core.*`` (and the openai_sdk submodules that
    captured it) so the next ``from agents.openai_sdk import agent``
    re-resolves ``from core import history`` against the openai_sdk
    package — not whatever autonomous-core another test left behind in
    ``sys.modules``. ``agents.openai_sdk.llm_client`` is preserved so
    ``@pytest.fixture mock_openai``'s ``patch(...)`` target keeps its
    identity across the fixture/test boundary.
    """
    for _k in list(sys.modules.keys()):
        if _k == "core" or _k.startswith("core."):
            del sys.modules[_k]
        if _k == "tools" or _k.startswith("tools."):
            del sys.modules[_k]
        if _k in (
            "agents.openai_sdk.agent",
            "agents.openai_sdk.prompts",
            "agents.openai_sdk.tools",
        ):
            del sys.modules[_k]


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def mock_openai(monkeypatch):
    """Patch the ``OpenAI`` class inside ``llm_client`` and reset the
    singleton cache so each test gets a clean stub."""
    _reset_core_resolution()
    _reset_llm_client_cache()
    monkeypatch.setenv("LLM_URL", "http://test/llm")
    monkeypatch.setenv("AGENT_TOKEN", "test-token")
    monkeypatch.setenv("MINER_UID", "42")
    with patch("agents.openai_sdk.llm_client.OpenAI") as m:
        yield m
    _reset_llm_client_cache()


def _make_completion(content: str = "", tool_calls=None, finish_reason="stop"):
    """Build an SDK-shaped ChatCompletion mock."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    # model_dump returns a plain dict (the SDK does this in production)
    payload = {"role": "assistant", "content": content}
    if tool_calls:
        payload["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
    msg.model_dump = MagicMock(return_value=payload)

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = (
        finish_reason if not tool_calls else "tool_calls"
    )

    completion = MagicMock()
    completion.choices = [choice]
    return completion


def _make_tool_call(name: str, arguments: dict, call_id: str = "call_1"):
    tc = MagicMock()
    tc.id = call_id
    tc.type = "function"
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


# ══════════════════════════════════════════════════════════════════
#  1. llm_client — caching & transient classification
# ══════════════════════════════════════════════════════════════════

class TestLlmClientCaching:
    def test_get_client_returns_cached_instance(self, mock_openai):
        from agents.openai_sdk.llm_client import get_client
        c1 = get_client()
        c2 = get_client()
        assert c1 is c2
        # OpenAI() should have been called exactly once
        assert mock_openai.call_count == 1

    def test_cache_invalidates_on_env_change(self, mock_openai, monkeypatch):
        from agents.openai_sdk.llm_client import get_client
        c1 = get_client()
        monkeypatch.setenv("LLM_URL", "http://different/llm")
        c2 = get_client()
        # New URL -> new client
        assert mock_openai.call_count == 2

    def test_get_client_passes_headers(self, mock_openai):
        from agents.openai_sdk.llm_client import get_client
        get_client()
        # OpenAI(...) was called with default_headers including the token
        kwargs = mock_openai.call_args.kwargs
        assert kwargs["base_url"].endswith("/v1")
        assert kwargs["default_headers"]["X-Agent-Token"] == "test-token"
        assert kwargs["default_headers"]["X-Miner-UID"] == "42"

    def test_get_client_prefers_arg_over_env(self, mock_openai, monkeypatch):
        """URL passed as an argument should take precedence over LLM_URL."""
        from agents.openai_sdk.llm_client import get_client
        monkeypatch.setenv("LLM_URL", "http://from-env/llm")
        get_client("http://from-arg/llm")
        kwargs = mock_openai.call_args.kwargs
        assert kwargs["base_url"] == "http://from-arg/llm/v1"

    def test_get_client_falls_back_to_env_when_no_arg(self, mock_openai):
        """If no arg is passed, the LLM_URL env var is still respected
        (backward compat for scripts/manual invocations)."""
        from agents.openai_sdk.llm_client import get_client
        get_client()
        kwargs = mock_openai.call_args.kwargs
        assert kwargs["base_url"] == "http://test/llm/v1"

    def test_get_client_raises_when_no_url_anywhere(
        self, mock_openai, monkeypatch
    ):
        """Neither arg nor env → RuntimeError with a clear message, not
        a raw KeyError that looks like a network failure."""
        _reset_llm_client_cache()
        monkeypatch.delenv("LLM_URL", raising=False)
        from agents.openai_sdk.llm_client import get_client
        with pytest.raises(RuntimeError) as exc:
            get_client()
        assert "LLM URL" in str(exc.value)


class TestTransientClassification:
    def test_timeout_is_transient(self):
        from openai import APITimeoutError
        from agents.openai_sdk.llm_client import _is_transient
        # APITimeoutError requires a request arg; supply a MagicMock
        err = APITimeoutError(MagicMock())
        assert _is_transient(err)

    def test_500_is_transient(self):
        from openai import APIError
        from agents.openai_sdk.llm_client import _is_transient
        err = APIError("server error", request=MagicMock(), body=None)
        err.status_code = 500
        assert _is_transient(err)

    def test_400_is_not_transient(self):
        from openai import APIError
        from agents.openai_sdk.llm_client import _is_transient
        err = APIError("bad request", request=MagicMock(), body=None)
        err.status_code = 400
        assert not _is_transient(err)

    def test_string_match_fallback(self):
        from agents.openai_sdk.llm_client import _is_transient
        assert _is_transient(RuntimeError("upstream returned 503"))
        assert not _is_transient(RuntimeError("schema validation failed"))


# ══════════════════════════════════════════════════════════════════
#  2. chat() — retry & failover
# ══════════════════════════════════════════════════════════════════

class TestChatRetry:
    def test_returns_completion_on_success(self, mock_openai):
        from agents.openai_sdk.llm_client import chat
        completion = _make_completion(content="hello")
        mock_openai.return_value.chat.completions.create.return_value = (
            completion
        )
        resp = chat(messages=[{"role": "user", "content": "hi"}])
        assert resp is completion

    def test_transient_then_success(self, mock_openai):
        """A transient error on the first call should be retried and
        the second call's success returned."""
        from openai import APITimeoutError
        from agents.openai_sdk.llm_client import chat
        completion = _make_completion(content="ok")
        mock_create = mock_openai.return_value.chat.completions.create
        mock_create.side_effect = [
            APITimeoutError(MagicMock()),
            completion,
        ]
        # base_delay=0 to keep the test fast
        resp = chat(
            messages=[{"role": "user", "content": "hi"}],
            base_delay=0.0,
        )
        assert resp is completion
        assert mock_create.call_count == 2

    def test_fails_fast_on_non_transient(self, mock_openai):
        """A 4xx non-transient error should raise on the first attempt
        without burning the retry budget."""
        from openai import APIError
        from agents.openai_sdk.llm_client import chat
        err = APIError("bad request", request=MagicMock(), body=None)
        err.status_code = 400
        mock_create = mock_openai.return_value.chat.completions.create
        mock_create.side_effect = err
        with pytest.raises(APIError):
            chat(
                messages=[{"role": "user", "content": "hi"}],
                base_delay=0.0,
            )
        assert mock_create.call_count == 1

    def test_model_pool_rotates(self, mock_openai):
        """Each retry should pick the next model in the pool."""
        from openai import APITimeoutError
        from agents.openai_sdk.llm_client import chat
        completion = _make_completion(content="ok")
        mock_create = mock_openai.return_value.chat.completions.create
        mock_create.side_effect = [
            APITimeoutError(MagicMock()),
            APITimeoutError(MagicMock()),
            completion,
        ]
        chat(
            messages=[{"role": "user", "content": "hi"}],
            model=["model-a", "model-b", "model-c"],
            base_delay=0.0,
            max_retries=3,
        )
        used_models = [
            call.kwargs["model"] for call in mock_create.call_args_list
        ]
        assert used_models == ["model-a", "model-b", "model-c"]

    def test_deadline_skips_backoff_sleep(self, mock_openai):
        """If the deadline would be exceeded by sleeping for backoff,
        chat() should bail out rather than waiting past the deadline."""
        from openai import APITimeoutError
        from agents.openai_sdk.llm_client import chat
        mock_create = mock_openai.return_value.chat.completions.create
        mock_create.side_effect = APITimeoutError(MagicMock())
        start = time.monotonic()
        with pytest.raises(APITimeoutError):
            chat(
                messages=[{"role": "user", "content": "hi"}],
                base_delay=10.0,         # would normally sleep for 10s
                max_retries=3,
                deadline=time.monotonic() + 1,  # only 1s left
            )
        elapsed = time.monotonic() - start
        # No way we slept 10s — the deadline cut off the backoff.
        assert elapsed < 5


# ══════════════════════════════════════════════════════════════════
#  3. tools — schema + handlers + circuit breaker
# ══════════════════════════════════════════════════════════════════

VALID_CODE = '''\
import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self, context_len, prediction_len, num_variates, n_q):
        super().__init__()
        self.pred = prediction_len
        self.nv = num_variates
        self.nq = n_q
        self.fc = nn.Linear(context_len, prediction_len * n_q)
    def forward(self, x):
        b, L, V = x.shape
        h = x.transpose(1, 2)
        h = self.fc(h)
        return h.view(b, V, self.pred, self.nq).permute(0, 2, 1, 3)

def build_model(context_len, prediction_len, num_variates, quantiles):
    return M(context_len, prediction_len, num_variates, len(quantiles))

def build_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)
'''


def _make_challenge(frontier=None, with_flops=True):
    ch = {
        "feasible_frontier": frontier or [],
        "task": {
            "name": "ts_forecasting",
            "description": "Time series forecasting",
            "task_params": {
                "context_len": 512,
                "prediction_len": 96,
                "num_variates": 1,
                "quantiles": [0.1, 0.5, 0.9],
            },
            "constraints": [
                "Output shape must be (batch, prediction_len, "
                "num_variates, len(quantiles))"
            ],
            "objectives": [{"name": "crps", "primary": True}],
        },
    }
    if with_flops:
        ch["min_flops_equivalent"] = 500_000
        ch["max_flops_equivalent"] = 2_000_000
    return ch


class TestToolSchema:
    def test_required_tools_present(self):
        from agents.openai_sdk.tools import TOOLS
        names = {t["function"]["name"] for t in TOOLS}
        # Full parity with the autonomous agent's tool surface, plus
        # the scratchpad-directory file tools.
        assert {
            "analyze_task", "validate_code", "estimate_flops",
            "size_to_flops",
            "list_frontier", "get_frontier_member", "submit",
            "search_papers", "query_db", "estimate_layer_flops",
            "sketch_architecture", "trace_architecture",
            "check_output_shape", "read_scratchpad", "write_scratchpad",
            "list_files", "read_file", "write_file", "search_files",
            "time_remaining",
        } == names

    def test_submit_requires_code_name_motivation(self):
        from agents.openai_sdk.tools import TOOLS
        submit = next(
            t for t in TOOLS if t["function"]["name"] == "submit"
        )
        required = submit["function"]["parameters"]["required"]
        assert {"code", "name", "motivation"} <= set(required)


class TestToolHandlers:
    def test_analyze_task_returns_json(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge())
        result = handlers["analyze_task"]()
        data = json.loads(result)
        assert data["name"] == "ts_forecasting"
        assert "context_len" in data["task_params"]

    def test_validate_code_ok(self):
        from agents.openai_sdk.tools import build_handlers
        # Drop FLOPs gate so VALID_CODE passes structural-only validation
        ch = _make_challenge(with_flops=False)
        handlers = build_handlers(ch)
        result = handlers["validate_code"](code=VALID_CODE)
        assert result == "ok"

    def test_validate_code_errors(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge())
        result = handlers["validate_code"](code="x = 1")
        assert result.startswith("errors:")
        assert "build_model" in result

    def test_list_frontier_empty(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge())
        result = handlers["list_frontier"]()
        assert "empty" in result.lower() or "bootstrapping" in result.lower()

    def test_list_frontier_with_data(self):
        from agents.openai_sdk.tools import build_handlers
        frontier = [{"name": "m1", "objectives": {"crps": 0.4}, "code": "x=1"}]
        handlers = build_handlers(_make_challenge(frontier=frontier))
        result = handlers["list_frontier"]()
        items = json.loads(result)
        assert items[0]["idx"] == 0
        assert items[0]["objectives"]["crps"] == 0.4

    def test_get_frontier_member_invalid_idx(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge(frontier=[{"name": "m1"}]))
        result = handlers["get_frontier_member"](idx=99)
        assert "out of range" in result

    def test_get_frontier_member_returns_code(self):
        from agents.openai_sdk.tools import build_handlers
        frontier = [{"name": "m1", "objectives": {"crps": 0.4}, "code": "x=1"}]
        handlers = build_handlers(_make_challenge(frontier=frontier))
        result = handlers["get_frontier_member"](idx=0)
        data = json.loads(result)
        assert data["code"] == "x=1"

    def test_submit_raises_signal(self):
        from agents.openai_sdk.tools import SubmitSignal, build_handlers
        handlers = build_handlers(_make_challenge())
        with pytest.raises(SubmitSignal) as excinfo:
            handlers["submit"](
                code=VALID_CODE, name="m", motivation="testing",
            )
        assert excinfo.value.code == VALID_CODE
        assert excinfo.value.name == "m"


class TestFileTools:
    def test_write_then_read_roundtrip(self, tmp_path):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(
            _make_challenge(), scratch_dir=str(tmp_path),
        )
        write_result = handlers["write_file"](
            name="design.md", content="# plan\nfoo",
        )
        assert "wrote design.md" in write_result
        read_result = handlers["read_file"](name="design.md")
        assert read_result == "# plan\nfoo"

    def test_list_files_excludes_state_json(self, tmp_path):
        from agents.openai_sdk.tools import build_handlers
        (tmp_path / "state.json").write_text("{}")
        handlers = build_handlers(
            _make_challenge(), scratch_dir=str(tmp_path),
        )
        handlers["write_file"](name="notes.md", content="hi")
        listed = json.loads(handlers["list_files"]())
        names = [item["name"] for item in listed]
        assert "notes.md" in names
        assert "state.json" not in names

    def test_list_files_empty(self, tmp_path):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(
            _make_challenge(), scratch_dir=str(tmp_path),
        )
        assert handlers["list_files"]() == "no files yet"

    def test_read_missing_file(self, tmp_path):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(
            _make_challenge(), scratch_dir=str(tmp_path),
        )
        result = handlers["read_file"](name="nope.md")
        assert "not found" in result

    def test_rejects_path_separators(self, tmp_path):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(
            _make_challenge(), scratch_dir=str(tmp_path),
        )
        for bad in ("../escape.md", "sub/file.md", "/abs.md", "..", "."):
            result = handlers["write_file"](name=bad, content="x")
            assert result.startswith("error:"), f"expected reject for {bad!r}"

    def test_rejects_reserved_name(self, tmp_path):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(
            _make_challenge(), scratch_dir=str(tmp_path),
        )
        result = handlers["write_file"](name="state.json", content="{}")
        assert "reserved" in result

    def test_enforces_size_cap(self, tmp_path):
        from agents.openai_sdk.tools import build_handlers, FILE_MAX_BYTES
        handlers = build_handlers(
            _make_challenge(), scratch_dir=str(tmp_path),
        )
        big = "x" * (FILE_MAX_BYTES + 1)
        result = handlers["write_file"](name="big.md", content=big)
        assert result.startswith("error:") and "max" in result

    def test_unavailable_without_scratch_dir(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge(), scratch_dir=None)
        assert "unavailable" in handlers["list_files"]()
        assert "unavailable" in handlers["read_file"](name="x.md")
        assert "unavailable" in handlers["write_file"](
            name="x.md", content="y",
        )

    def test_search_files_finds_matches(self, tmp_path):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(
            _make_challenge(), scratch_dir=str(tmp_path),
        )
        handlers["write_file"](
            name="design.md",
            content="# Plan\nUse TRANSFORMER backbone\nkeep it tiny",
        )
        handlers["write_file"](
            name="notes.md", content="no hits here",
        )
        result = handlers["search_files"](query="transformer")
        assert "[design.md]" in result
        assert "transformer" in result.lower()
        assert "[notes.md]" not in result

    def test_search_files_no_match(self, tmp_path):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(
            _make_challenge(), scratch_dir=str(tmp_path),
        )
        handlers["write_file"](name="a.md", content="hello world")
        result = handlers["search_files"](query="nonexistent")
        assert "no matches" in result

    def test_search_files_empty_query(self, tmp_path):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(
            _make_challenge(), scratch_dir=str(tmp_path),
        )
        result = handlers["search_files"](query="")
        assert result.startswith("error:")

    def test_overwrite_does_not_hit_count_limit(self, tmp_path):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(
            _make_challenge(), scratch_dir=str(tmp_path),
        )
        handlers["write_file"](name="a.md", content="one")
        # Overwriting the same file should always succeed regardless
        # of the file-count cap.
        result = handlers["write_file"](name="a.md", content="two")
        assert "wrote a.md" in result
        assert handlers["read_file"](name="a.md") == "two"


SIZED_CODE_TEMPLATE = '''\
import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self, context_len, prediction_len, num_variates, n_q, h):
        super().__init__()
        self.pred = prediction_len
        self.nv = num_variates
        self.nq = n_q
        self.enc = nn.Linear(context_len, h)
        self.dec = nn.Linear(h, prediction_len * n_q)
    def forward(self, x):
        b, L, V = x.shape
        h = x.transpose(1, 2)
        h = self.enc(h)
        h = self.dec(h)
        return h.view(b, V, self.pred, self.nq).permute(0, 2, 1, 3)

def build_model(context_len, prediction_len, num_variates, quantiles):
    return M(context_len, prediction_len, num_variates, len(quantiles), {{SIZE}})

def build_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)
'''


class TestSizeToFlops:
    def test_missing_placeholder_errors(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge())
        result = handlers["size_to_flops"](
            code_template="def build_model(**kw): pass",
            size_min=1, size_max=100,
        )
        assert "SIZE" in result and "error" in result.lower()

    def test_empty_template_errors(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge())
        result = handlers["size_to_flops"](
            code_template="", size_min=1, size_max=100,
        )
        assert result.startswith("error:")

    def test_invalid_range_errors(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge())
        result = handlers["size_to_flops"](
            code_template="# {{SIZE}}", size_min=100, size_max=10,
        )
        assert "error" in result.lower()

    def test_zero_size_min_errors(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge())
        result = handlers["size_to_flops"](
            code_template="# {{SIZE}}", size_min=0, size_max=100,
        )
        assert "error" in result.lower()

    def test_no_target_and_no_budget_errors(self):
        from agents.openai_sdk.tools import build_handlers
        # Challenge without FLOPs budget and no target_flops passed
        ch = _make_challenge(with_flops=False)
        handlers = build_handlers(ch)
        result = handlers["size_to_flops"](
            code_template="# {{SIZE}}", size_min=1, size_max=100,
        )
        assert "error" in result.lower()
        assert "target" in result.lower() or "budget" in result.lower()

    def test_happy_path_finds_best(self):
        """End-to-end: sweep a real linear build_model and find a size
        whose FLOPs are near the target. Kept small (size range 8..64)
        so it runs fast inside CI."""
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge())
        # target_flops defaults to 60% of max (=1.2M). For the template
        # above, tiny hidden sizes (8..64) give ~50K..400K FLOPs which
        # is well below target — so "best" is the highest size probed.
        result = handlers["size_to_flops"](
            code_template=SIZED_CODE_TEMPLATE,
            size_min=8, size_max=64,
            target_flops=200_000,
        )
        assert "best:" in result
        assert "size=" in result
        assert "flops=" in result
        # Should mention it did some probes
        assert "probes:" in result


class TestScratchpadNotes:
    """Covers the structured write_scratchpad / read_scratchpad contract.

    ``write_scratchpad`` must route ``hypothesis`` / ``dead_end`` +
    ``reason`` / ``observation`` into the matching notes section, still
    honour the deprecated ``notes`` string for in-flight rounds, and
    reject empty calls. ``read_scratchpad`` must render the structured
    sections so the next round can see them.
    """

    def _handlers_with_state(self, state=None):
        from agents.openai_sdk.tools import build_handlers
        state = {} if state is None else state
        handlers = build_handlers(_make_challenge(), state=state)
        return handlers, state

    def test_write_hypothesis_appends_to_section(self):
        handlers, state = self._handlers_with_state()
        out = handlers["write_scratchpad"](hypothesis="try SSM backbone")
        assert "hypothesis" in out
        assert state["notes"]["open_hypotheses"] == ["try SSM backbone"]

    def test_write_dead_end_with_reason_combines_them(self):
        handlers, state = self._handlers_with_state()
        out = handlers["write_scratchpad"](
            dead_end="plain MLP", reason="below FLOPs gate",
        )
        assert "dead_end" in out
        entry = state["notes"]["dead_ends"][0]
        assert "plain MLP" in entry
        assert "below FLOPs gate" in entry

    def test_write_dead_end_without_reason_still_stored(self):
        handlers, state = self._handlers_with_state()
        handlers["write_scratchpad"](dead_end="untried idea")
        assert state["notes"]["dead_ends"] == ["untried idea"]

    def test_write_observation_appends_to_section(self):
        handlers, state = self._handlers_with_state()
        handlers["write_scratchpad"](observation="output is (B, N, K)")
        assert state["notes"]["task_observations"] == [
            "output is (B, N, K)",
        ]

    def test_multiple_fields_in_one_call(self):
        handlers, state = self._handlers_with_state()
        out = handlers["write_scratchpad"](
            hypothesis="try attention",
            dead_end="pure Linear", reason="underfits",
            observation="input channels are fixed",
        )
        for tag in ("hypothesis", "dead_end", "observation"):
            assert tag in out
        assert state["notes"]["open_hypotheses"] == ["try attention"]
        assert "underfits" in state["notes"]["dead_ends"][0]
        assert state["notes"]["task_observations"] == [
            "input channels are fixed",
        ]

    def test_deprecated_notes_field_still_accepted(self):
        handlers, state = self._handlers_with_state()
        out = handlers["write_scratchpad"](notes="legacy free-form text")
        assert "notes(deprecated)" in out or "deprecated" in out
        assert state.get("agent_notes") == "legacy free-form text"

    def test_empty_call_returns_error(self):
        handlers, _ = self._handlers_with_state()
        out = handlers["write_scratchpad"]()
        assert out.startswith("error:")

    def test_whitespace_only_values_return_error(self):
        handlers, _ = self._handlers_with_state()
        out = handlers["write_scratchpad"](
            hypothesis="   ", dead_end="", observation="",
        )
        assert out.startswith("error:")

    def test_cap_enforced_through_handler(self):
        from core.history import NOTES_MAX_ENTRIES
        handlers, state = self._handlers_with_state()
        for i in range(NOTES_MAX_ENTRIES + 3):
            handlers["write_scratchpad"](hypothesis=f"h{i}")
        bucket = state["notes"]["open_hypotheses"]
        assert len(bucket) == NOTES_MAX_ENTRIES
        assert bucket[-1] == f"h{NOTES_MAX_ENTRIES + 2}"

    def test_wrote_this_round_flag_set_after_success(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge(), state={})
        # state_holder is stashed on the submit wrapper
        state_holder = handlers["submit"]._state_holder
        assert state_holder["wrote_this_round"] is False
        handlers["write_scratchpad"](observation="anything")
        assert state_holder["wrote_this_round"] is True

    def test_wrote_this_round_stays_false_on_error(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge(), state={})
        state_holder = handlers["submit"]._state_holder
        handlers["write_scratchpad"]()  # empty — error path
        assert state_holder["wrote_this_round"] is False

    def test_read_scratchpad_empty_state(self):
        handlers, _ = self._handlers_with_state()
        out = handlers["read_scratchpad"]()
        assert "first round" in out or "empty" in out

    def test_read_scratchpad_renders_structured_notes(self):
        handlers, _ = self._handlers_with_state()
        handlers["write_scratchpad"](hypothesis="attn variant A")
        handlers["write_scratchpad"](
            dead_end="dense MLP", reason="overshoots FLOPs",
        )
        out = handlers["read_scratchpad"]()
        assert "Open Hypotheses" in out
        assert "attn variant A" in out
        assert "Dead Ends" in out
        assert "dense MLP" in out

    def test_read_scratchpad_shows_legacy_notes_section(self):
        handlers, _ = self._handlers_with_state()
        handlers["write_scratchpad"](notes="legacy blob")
        out = handlers["read_scratchpad"]()
        assert "legacy blob" in out


class TestCallCounts:
    """Per-round tool-usage telemetry is stashed on the submit
    wrapper as ``_call_counts``. Each wrapped handler invocation bumps
    its entry; the agent logs the summary at end-of-round."""

    def test_counter_starts_empty(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge())
        assert handlers["submit"]._call_counts == {}

    def test_counter_bumps_on_each_call(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge())
        counts = handlers["submit"]._call_counts
        handlers["analyze_task"]()
        handlers["analyze_task"]()
        handlers["list_frontier"]()
        assert counts == {"analyze_task": 2, "list_frontier": 1}

    def test_counter_bumps_even_on_error_path(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge())
        counts = handlers["submit"]._call_counts
        handlers["get_frontier_member"](idx=99)   # error
        handlers["get_frontier_member"](idx=99)   # error
        assert counts["get_frontier_member"] == 2


class TestCircuitBreaker:
    def test_trips_after_repeated_identical_errors(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge(frontier=[]))
        r1 = handlers["get_frontier_member"](idx=99)
        r2 = handlers["get_frontier_member"](idx=99)
        r3 = handlers["get_frontier_member"](idx=99)
        assert r1 == r2
        assert "circuit open" in r3.lower()

    def test_resets_after_non_error(self):
        from agents.openai_sdk.tools import build_handlers
        # Long, JSON list result → not error-shaped → counter resets
        frontier = [{"name": "m" * 200, "objectives": {"crps": 0.4},
                     "code": "x" * 200}]
        handlers = build_handlers(_make_challenge(frontier=frontier))
        handlers["get_frontier_member"](idx=99)  # error
        handlers["get_frontier_member"](idx=99)  # error #2
        good = handlers["get_frontier_member"](idx=0)  # long ok response
        assert "circuit open" not in good.lower()
        # Counter reset → next bad call is just a regular error
        bad = handlers["get_frontier_member"](idx=99)
        assert "circuit open" not in bad.lower()


# ══════════════════════════════════════════════════════════════════
#  4. agent.design_architecture — end-to-end flow with mocks
# ══════════════════════════════════════════════════════════════════

class TestDesignArchitecture:
    def test_submit_on_first_turn(self, mock_openai):
        """LLM submits validated code on the first turn — agent ships it."""
        from agents.openai_sdk import agent

        tc = _make_tool_call("submit", {
            "code": VALID_CODE,
            "name": "instant",
            "motivation": "first-shot",
        })
        completion = _make_completion(tool_calls=[tc])
        mock_openai.return_value.chat.completions.create.return_value = (
            completion
        )

        ch = _make_challenge(with_flops=False)
        ch["agent_seconds"] = 60
        result = agent.design_architecture(ch, gated_client=None)
        assert result["code"] == VALID_CODE
        assert result["name"] == "instant"

    def test_falls_back_to_template_when_no_llm_code(self, mock_openai):
        """When the LLM never produces code, the agent ships the
        guaranteed-valid fallback template."""
        from agents.openai_sdk import agent

        # Always return plain text — no tool calls, no code blocks
        completion = _make_completion(content="thinking...")
        mock_openai.return_value.chat.completions.create.return_value = (
            completion
        )

        ch = _make_challenge()
        ch["agent_seconds"] = 60
        result = agent.design_architecture(ch, gated_client=None)
        assert "fallback" in result["name"]
        assert "build_model" in result["code"]
        assert "build_optimizer" in result["code"]

    def test_chat_failure_uses_fallback(self, mock_openai):
        """If every LLM call raises, the agent still ships fallback code."""
        from openai import APIError
        from agents.openai_sdk import agent
        err = APIError("bad", request=MagicMock(), body=None)
        err.status_code = 400  # non-transient → no retries inside chat()
        mock_openai.return_value.chat.completions.create.side_effect = err

        ch = _make_challenge()
        ch["agent_seconds"] = 60
        result = agent.design_architecture(ch, gated_client=None)
        # Fallback path → bucket-tagged template name
        assert "fallback" in result["name"]
        assert result["code"] != ""
        # Honest failure motivation — not the generic template message.
        assert "LLM chat failed" in result["motivation"]

    def test_missing_llm_url_shows_config_error_motivation(
        self, mock_openai, monkeypatch
    ):
        """With no challenge['llm_url'] and no LLM_URL env var, the
        startup check should trip, skip both phases, and ship the
        fallback with a config-error motivation (not the old generic
        'could not produce code' message)."""
        _reset_llm_client_cache()
        monkeypatch.delenv("LLM_URL", raising=False)
        from agents.openai_sdk import agent

        ch = _make_challenge()
        ch["agent_seconds"] = 60
        # No llm_url key in challenge
        result = agent.design_architecture(ch, gated_client=None)

        assert "fallback" in result["name"]
        assert result["code"] != ""
        assert "config error" in result["motivation"].lower()
        # The chat client should never have been built or called.
        assert not mock_openai.return_value.chat.completions.create.called

    def test_challenge_llm_url_is_used(self, mock_openai, monkeypatch):
        """challenge['llm_url'] should be preferred over the env var —
        the harness passes the URL through the challenge dict."""
        _reset_llm_client_cache()
        monkeypatch.delenv("LLM_URL", raising=False)
        from agents.openai_sdk import agent

        tc = _make_tool_call("submit", {
            "code": VALID_CODE,
            "name": "from-challenge",
            "motivation": "used challenge.llm_url",
        })
        completion = _make_completion(tool_calls=[tc])
        mock_openai.return_value.chat.completions.create.return_value = (
            completion
        )

        ch = _make_challenge(with_flops=False)
        ch["agent_seconds"] = 60
        ch["llm_url"] = "http://from-challenge/llm"
        result = agent.design_architecture(ch, gated_client=None)

        assert result["name"] == "from-challenge"
        # OpenAI was built with the challenge-supplied URL.
        kwargs = mock_openai.call_args.kwargs
        assert kwargs["base_url"] == "http://from-challenge/llm/v1"


# ══════════════════════════════════════════════════════════════════
#  5. _extract_code_block helper
# ══════════════════════════════════════════════════════════════════

class TestExtractCodeBlock:
    def test_finds_python_fence(self):
        from agents.openai_sdk.agent import _extract_code_block
        text = "Here is code:\n```python\nprint('hi')\n```\nThanks."
        assert _extract_code_block(text) == "print('hi')"

    def test_returns_empty_when_no_block(self):
        from agents.openai_sdk.agent import _extract_code_block
        assert _extract_code_block("just prose") == ""

    def test_handles_bare_fence(self):
        from agents.openai_sdk.agent import _extract_code_block
        text = "```\nprint('hi')\n```"
        assert _extract_code_block(text) == "print('hi')"
