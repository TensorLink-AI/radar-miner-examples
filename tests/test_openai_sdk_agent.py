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


def _make_challenge(frontier=None, with_flops=True, cognition_wiki_url=""):
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
        "cognition_wiki_url": cognition_wiki_url,
    }
    if with_flops:
        ch["min_flops_equivalent"] = 500_000
        ch["max_flops_equivalent"] = 2_000_000
    return ch


class _MockClient:
    """Minimal GatedClient stub for wiki fetch tests.

    Records call counts and returns a canned ``get`` response. Other
    methods raise so tests don't accidentally rely on them.
    """

    def __init__(self, get_response: bytes = b""):
        self._get_response = get_response
        self.get_call_count = 0

    def get(self, url, **_kwargs):
        # ``call_with_timeout`` forwards a ``timeout=`` kwarg — accept it.
        self.get_call_count += 1
        return self._get_response


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
            "check_output_shape", "read_scratchpad",
            "read_my_submissions", "write_scratchpad",
            "list_files", "read_file", "write_file", "search_files",
            "time_remaining",
            "cognition_wiki_index", "cognition_wiki_read",
        } == names

    def test_submit_requires_name_motivation(self):
        from agents.openai_sdk.tools import TOOLS
        submit = next(
            t for t in TOOLS if t["function"]["name"] == "submit"
        )
        required = submit["function"]["parameters"]["required"]
        # ``code`` is conditionally required (only when no
        # ``candidate_id`` is supplied), so it's not in the JSON-schema
        # required list — the handler enforces "one of code/candidate_id"
        # at runtime.
        assert {"name", "motivation"} <= set(required)


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
        # Success result starts with "ok" and includes a submit-now
        # directive so the LLM doesn't go back to sketching.
        assert result.startswith("ok")
        assert "submit" in result.lower()

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

    def test_submit_raises_signal_after_scratchpad_write(self):
        """Happy path — note is written, submit raises SubmitSignal."""
        from agents.openai_sdk.tools import SubmitSignal, build_handlers
        handlers = build_handlers(_make_challenge())
        handlers["write_scratchpad"](observation="tried variant A")
        with pytest.raises(SubmitSignal) as excinfo:
            handlers["submit"](
                code=VALID_CODE, name="m", motivation="testing",
            )
        assert excinfo.value.code == VALID_CODE
        assert excinfo.value.name == "m"

    def test_submit_raises_signal_without_scratchpad_note(self, capsys):
        """No-note path — submit no longer blocks. SubmitSignal still
        fires, with one stderr warning line for diagnostics."""
        from agents.openai_sdk.tools import SubmitSignal, build_handlers
        handlers = build_handlers(_make_challenge())
        with pytest.raises(SubmitSignal) as excinfo:
            handlers["submit"](
                code=VALID_CODE, name="m", motivation="no note",
            )
        assert excinfo.value.code == VALID_CODE
        captured = capsys.readouterr()
        assert "submit without scratchpad note this round" in captured.err


class TestCandidateLineage:
    """Feature 1: code lineage / candidate IDs.

    A candidate is keyed by ``cand_<8 hex>``, derived from a sha1 of the
    code, so identical code re-yields the same id (natural dedup). Each
    record holds code, flops, trace, validated, submitted, created_at.
    """

    def _state(self, handlers):
        return handlers["submit"]._state_holder["state"]

    def test_sketch_returns_candidate_id_and_stores_record(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge(with_flops=False))
        sketch_code = (
            "import torch.nn as nn\n"
            "def build_model(context_len, prediction_len, "
            "num_variates, quantiles):\n"
            "    return nn.Linear(context_len, prediction_len)\n"
        )
        result = handlers["sketch_architecture"](code=sketch_code)
        # Output ends with a candidate_id line in cand_<hex> form.
        last = [l for l in result.splitlines() if l.strip()][-1]
        assert last.startswith("candidate_id: cand_")
        cid = last.split(": ", 1)[1]
        assert len(cid) == len("cand_") + 8
        # Record is stored with the code, validated/submitted False.
        cands = self._state(handlers)["candidates"]
        assert cid in cands
        record = cands[cid]
        assert "def build_model" in record["code"]
        assert record["validated"] is False
        assert record["submitted"] is False
        assert "created_at" in record

    def test_sketch_dedupes_identical_code(self):
        """Same code → same id; the dict still has one entry."""
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge(with_flops=False))
        code = (
            "import torch.nn as nn\n"
            "def build_model(context_len, prediction_len, "
            "num_variates, quantiles):\n"
            "    return nn.Linear(context_len, prediction_len)\n"
        )
        r1 = handlers["sketch_architecture"](code=code)
        r2 = handlers["sketch_architecture"](code=code)
        cid1 = r1.splitlines()[-1].split(": ", 1)[1]
        cid2 = r2.splitlines()[-1].split(": ", 1)[1]
        assert cid1 == cid2
        assert len(self._state(handlers)["candidates"]) == 1

    def test_validate_with_candidate_id_pulls_from_state(self):
        """validate_code with candidate_id ignores the code arg and
        loads the source from state."""
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge(with_flops=False))
        # Stash VALID_CODE under a known id by sketching it first.
        sketch_result = handlers["sketch_architecture"](code=VALID_CODE)
        cid = sketch_result.splitlines()[-1].split(": ", 1)[1]
        # Pass NO code, only the id — the handler must look up the
        # source from state.
        result = handlers["validate_code"](candidate_id=cid)
        assert result.startswith("ok")
        record = self._state(handlers)["candidates"][cid]
        assert record["validated"] is True

    def test_validate_with_unknown_candidate_id_errors(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge(with_flops=False))
        result = handlers["validate_code"](candidate_id="cand_deadbeef")
        assert result.startswith("errors:")
        assert "cand_deadbeef" in result
        assert "not found" in result

    def test_validate_without_candidate_id_autogenerates_on_success(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge(with_flops=False))
        result = handlers["validate_code"](code=VALID_CODE)
        assert result.startswith("ok")
        last = [l for l in result.splitlines() if l.strip()][-1]
        assert last.startswith("candidate_id: cand_")
        cid = last.split(": ", 1)[1]
        cands = self._state(handlers)["candidates"]
        assert cid in cands
        assert cands[cid]["validated"] is True

    def test_validate_failure_does_not_create_candidate(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge())
        result = handlers["validate_code"](code="x = 1")
        assert result.startswith("errors:")
        assert self._state(handlers).get("candidates", {}) == {}

    def test_submit_with_candidate_id_ships_from_state(self):
        from agents.openai_sdk.tools import SubmitSignal, build_handlers
        handlers = build_handlers(_make_challenge(with_flops=False))
        handlers["validate_code"](code=VALID_CODE)
        # The validate_code success line carries the id; pull it out.
        # (Equivalent to the LLM reading the tool result.)
        cands = self._state(handlers)["candidates"]
        cid = next(iter(cands))
        handlers["write_scratchpad"](observation="ok")
        with pytest.raises(SubmitSignal) as excinfo:
            handlers["submit"](
                candidate_id=cid, name="m", motivation="from id",
            )
        # The shipped code came from state, not from a code= arg.
        assert excinfo.value.code == cands[cid]["code"]
        assert cands[cid]["submitted"] is True

    def test_submit_with_unknown_candidate_id_errors(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge())
        handlers["write_scratchpad"](observation="ok")
        result = handlers["submit"](
            candidate_id="cand_deadbeef", name="m", motivation="x",
        )
        assert "not found" in result

    def test_submit_without_candidate_id_behaves_as_before(self):
        """Backwards-compat: submit(code=..., name=..., motivation=...)
        still works and raises SubmitSignal with the supplied code."""
        from agents.openai_sdk.tools import SubmitSignal, build_handlers
        handlers = build_handlers(_make_challenge())
        handlers["write_scratchpad"](observation="ok")
        with pytest.raises(SubmitSignal) as excinfo:
            handlers["submit"](
                code=VALID_CODE, name="m", motivation="legacy",
            )
        assert excinfo.value.code == VALID_CODE

    def test_submit_with_neither_code_nor_id_returns_error(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge())
        handlers["write_scratchpad"](observation="ok")
        result = handlers["submit"](name="m", motivation="empty")
        assert result.startswith("error:")
        assert "candidate_id" in result or "code" in result

    def test_candidates_survive_save_load_roundtrip(self, tmp_path):
        """A candidate written via the handler persists through
        history.save_state/load_state without losing fields."""
        import sys
        sys.path.insert(0, "agents/openai_sdk")
        for k in list(sys.modules):
            if k == "core" or k.startswith("core."):
                del sys.modules[k]
        from core.history import (  # noqa: E402
            load_state, save_state,
        )
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(
            _make_challenge(with_flops=False),
            scratch_dir=str(tmp_path),
        )
        handlers["sketch_architecture"](code=VALID_CODE)
        state = self._state(handlers)
        save_state(str(tmp_path), state)
        reloaded = load_state(str(tmp_path))
        assert "candidates" in reloaded
        assert reloaded["candidates"] == state["candidates"]
        # Spot-check the record shape survived JSON round-trip.
        record = next(iter(reloaded["candidates"].values()))
        for key in ("code", "flops", "trace", "validated",
                    "submitted", "created_at"):
            assert key in record

    def test_legacy_state_without_candidates_loads_cleanly(self, tmp_path):
        """An old-schema state.json (no candidates key) must load and
        the handlers must treat the candidate dict as empty."""
        import json
        legacy = {
            "history": [{"name": "old", "code_hash": 123}],
            "notes": {"open_hypotheses": ["try X"]},
        }
        (tmp_path / "state.json").write_text(json.dumps(legacy))
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(
            _make_challenge(with_flops=False),
            scratch_dir=str(tmp_path),
        )
        state = self._state(handlers)
        # Legacy keys preserved.
        assert state["history"][0]["name"] == "old"
        # candidates absent on disk → handler-side accessors see {}.
        # (The dict is created lazily on first write.)
        assert state.get("candidates", {}) == {}
        # First sketch creates the dict without crashing on the absent
        # key.
        result = handlers["sketch_architecture"](code=VALID_CODE)
        assert "candidate_id: cand_" in result
        assert len(state["candidates"]) == 1


class TestReadMySubmissions:
    """Feature 2: read_my_submissions tool.

    submit appends a record to ``state["submissions"]`` with the full
    code blob; read_my_submissions surfaces the n most recent (newest
    first) and merge_results_into_state attaches score/rank when the
    next round's previous_results arrive.
    """

    def _state(self, handlers):
        return handlers["submit"]._state_holder["state"]

    def _ship(self, handlers, code, name, motivation, candidate_id=""):
        from agents.openai_sdk.tools import SubmitSignal
        handlers["write_scratchpad"](observation="ok")
        kwargs = {"code": code, "name": name, "motivation": motivation}
        if candidate_id:
            kwargs["candidate_id"] = candidate_id
        try:
            handlers["submit"](**kwargs)
        except SubmitSignal:
            pass

    def test_submit_records_into_state_submissions(self):
        from agents.openai_sdk.tools import build_handlers
        ch = _make_challenge()
        ch["round_id"] = 7
        handlers = build_handlers(ch)
        self._ship(handlers, VALID_CODE, "m", "first try")
        subs = self._state(handlers)["submissions"]
        assert len(subs) == 1
        rec = subs[0]
        assert rec["code"] == VALID_CODE
        assert rec["name"] == "m"
        assert rec["motivation"] == "first try"
        assert rec["round_id"] == "7"
        assert rec["score"] is None
        assert rec["rank"] is None
        assert rec["candidate_id"] is None
        assert "submitted_at" in rec
        assert "code_hash" in rec

    def test_submit_records_candidate_id_when_passed(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge(with_flops=False))
        validate_result = handlers["validate_code"](code=VALID_CODE)
        cid = [
            l for l in validate_result.splitlines()
            if l.startswith("candidate_id: ")
        ][-1].split(": ", 1)[1]
        self._ship(handlers, "", "m", "from id", candidate_id=cid)
        subs = self._state(handlers)["submissions"]
        assert subs[-1]["candidate_id"] == cid
        assert subs[-1]["code"] == VALID_CODE

    def test_read_returns_pending_when_no_score_yet(self):
        from agents.openai_sdk.tools import build_handlers
        ch = _make_challenge()
        ch["round_id"] = 12
        handlers = build_handlers(ch)
        self._ship(handlers, VALID_CODE, "wide_mlp", "wider hidden")
        out = handlers["read_my_submissions"](n=1)
        assert "round 12" in out
        assert "wide_mlp" in out
        assert "score: pending" in out
        assert "wider hidden" in out
        # Full code is shown when n=1.
        assert "import torch" in out
        assert "build_model" in out

    def test_read_truncates_when_n_gt_1(self):
        """A long-code submission must be truncated when n>1."""
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge())
        long_code = (
            "import torch\n" + "\n".join(
                f"x_{i} = {i}" for i in range(100)
            ) + "\n"
        )
        self._ship(handlers, long_code, "a", "first")
        self._ship(handlers, VALID_CODE, "b", "second")
        out = handlers["read_my_submissions"](n=2)
        assert "Submission 1 of 2" in out
        assert "Submission 2 of 2" in out
        # Newest is "b" — appears first in the rendered output.
        assert out.index("name=b") < out.index("name=a")
        # The long code (now in entry #2) must show the truncation
        # marker; VALID_CODE in entry #1 is short and stays intact.
        assert "more lines truncated" in out

    def test_read_full_code_when_n_eq_1(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge())
        long_code = (
            "import torch\n" + "\n".join(
                f"x_{i} = {i}" for i in range(100)
            ) + "\n"
        )
        self._ship(handlers, long_code, "long", "filler")
        out = handlers["read_my_submissions"](n=1)
        # No truncation marker, every line present.
        assert "more lines truncated" not in out
        assert "x_99 = 99" in out

    def test_read_empty_state(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge())
        assert handlers["read_my_submissions"]() == "no submissions yet"

    def test_read_default_n_is_3(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge())
        for i in range(5):
            self._ship(
                handlers, VALID_CODE.replace("Adam", f"Adam  # {i}"),
                f"v{i}", f"motivation {i}",
            )
        out = handlers["read_my_submissions"]()
        # Default 3, newest first → v4, v3, v2 visible; v1, v0 not.
        assert "name=v4" in out
        assert "name=v3" in out
        assert "name=v2" in out
        assert "name=v1" not in out
        assert "name=v0" not in out

    def test_read_clamps_n_to_at_least_1(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge())
        self._ship(handlers, VALID_CODE, "only", "only one")
        out = handlers["read_my_submissions"](n=0)
        assert "Submission 1 of 1" in out

    def test_legacy_state_loads_with_empty_submissions(self, tmp_path):
        """Old state.json without ``submissions`` key loads cleanly."""
        import json
        legacy = {
            "history": [{"name": "old", "code_hash": 123}],
            "notes": {"open_hypotheses": ["try X"]},
        }
        (tmp_path / "state.json").write_text(json.dumps(legacy))
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(
            _make_challenge(), scratch_dir=str(tmp_path),
        )
        assert handlers["read_my_submissions"]() == "no submissions yet"
        # First submit lazily creates the list.
        self._ship(handlers, VALID_CODE, "n", "m")
        assert len(self._state(handlers)["submissions"]) == 1

    def test_score_updated_when_previous_results_arrive(self):
        """Round N submits; round N+1's previous_results carries the
        score; merge_results_into_state must attach it to the
        submissions entry by code_hash."""
        from agents.openai_sdk.tools import build_handlers
        from agents.openai_sdk.core.history import (
            merge_results_into_state,
        )
        ch = _make_challenge()
        ch["round_id"] = 5
        handlers = build_handlers(ch)
        self._ship(handlers, VALID_CODE, "v1", "m1")
        state = self._state(handlers)
        ch_value = state["submissions"][0]["code_hash"]
        merge_results_into_state(state, [{
            "round_id": 5, "code_hash": ch_value,
            "score": 0.123, "rank": 3, "rank_total": 12,
        }])
        # Score / rank propagated to the submission record.
        rec = state["submissions"][0]
        assert rec["score"] == 0.123
        assert rec["rank"] == 3
        assert rec["rank_total"] == 12
        assert rec["scored_round_id"] == 5
        # And read_my_submissions renders them.
        out = handlers["read_my_submissions"](n=1)
        assert "score: 0.123 (rank 3/12)" in out

    def test_submissions_survive_save_load_roundtrip(self, tmp_path):
        import sys
        sys.path.insert(0, "agents/openai_sdk")
        for k in list(sys.modules):
            if k == "core" or k.startswith("core."):
                del sys.modules[k]
        from core.history import (  # noqa: E402
            load_state, save_state,
        )
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(
            _make_challenge(), scratch_dir=str(tmp_path),
        )
        self._ship(handlers, VALID_CODE, "n", "m")
        save_state(str(tmp_path), self._state(handlers))
        reloaded = load_state(str(tmp_path))
        assert "submissions" in reloaded
        assert reloaded["submissions"][0]["code"] == VALID_CODE
        for key in ("code", "code_hash", "name", "motivation",
                    "candidate_id", "round_id", "score", "rank",
                    "rank_total", "submitted_at"):
            assert key in reloaded["submissions"][0]


class TestHypothesisLinkage:
    """Feature 3: hypothesis → candidate_id → outcome linkage.

    write_scratchpad(hypothesis=..., candidate_id=...) attaches the id
    to the hypothesis record. read_scratchpad renders each link's
    current candidate state (validated/submitted) and score (when
    previous_results has merged it onto the matching submission).
    A summary line at the top gives the agent a quick situational read.
    """

    def _state(self, handlers):
        return handlers["submit"]._state_holder["state"]

    def _ship(self, handlers, code, name, motivation, candidate_id=""):
        from agents.openai_sdk.tools import SubmitSignal
        kwargs = {"code": code, "name": name, "motivation": motivation}
        if candidate_id:
            kwargs["candidate_id"] = candidate_id
        try:
            handlers["submit"](**kwargs)
        except SubmitSignal:
            pass

    def test_hypothesis_with_candidate_id_stores_linkage(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge())
        handlers["write_scratchpad"](
            hypothesis="try patch sizes", candidate_id="cand_a3f24c1d",
        )
        bucket = self._state(handlers)["notes"]["open_hypotheses"]
        assert len(bucket) == 1
        assert bucket[0]["text"] == "try patch sizes"
        assert bucket[0]["candidate_ids"] == ["cand_a3f24c1d"]

    def test_hypothesis_without_candidate_id_starts_empty_links(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge())
        handlers["write_scratchpad"](hypothesis="something to try")
        bucket = self._state(handlers)["notes"]["open_hypotheses"]
        assert bucket[0]["candidate_ids"] == []

    def test_duplicate_text_appends_candidate_id(self):
        """Re-stating the same hypothesis with a new id appends rather
        than creating a duplicate row."""
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge())
        handlers["write_scratchpad"](
            hypothesis="try patches", candidate_id="cand_aaaaaaaa",
        )
        handlers["write_scratchpad"](
            hypothesis="try patches", candidate_id="cand_bbbbbbbb",
        )
        bucket = self._state(handlers)["notes"]["open_hypotheses"]
        assert len(bucket) == 1
        assert bucket[0]["candidate_ids"] == [
            "cand_aaaaaaaa", "cand_bbbbbbbb",
        ]

    def test_duplicate_candidate_id_does_not_double_append(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge())
        handlers["write_scratchpad"](
            hypothesis="try patches", candidate_id="cand_aaaaaaaa",
        )
        handlers["write_scratchpad"](
            hypothesis="try patches", candidate_id="cand_aaaaaaaa",
        )
        bucket = self._state(handlers)["notes"]["open_hypotheses"]
        assert bucket[0]["candidate_ids"] == ["cand_aaaaaaaa"]

    def test_legacy_string_hypotheses_auto_upgrade_on_access(
        self, tmp_path,
    ):
        """Old scratchpads with bare-string hypotheses must be wrapped
        in dicts on first access — no manual migration step."""
        import json
        legacy = {
            "notes": {
                "open_hypotheses": ["try X", "try Y"],
                "dead_ends": [],
                "task_observations": [],
            },
        }
        (tmp_path / "state.json").write_text(json.dumps(legacy))
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(
            _make_challenge(), scratch_dir=str(tmp_path),
        )
        # Trigger an access through write_scratchpad — _notes() runs
        # the upgrade.
        handlers["write_scratchpad"](hypothesis="try Z")
        bucket = self._state(handlers)["notes"]["open_hypotheses"]
        assert all(isinstance(h, dict) for h in bucket)
        texts = [h["text"] for h in bucket]
        assert texts == ["try X", "try Y", "try Z"]
        # Upgraded entries get an empty candidate_ids list and no
        # created_at timestamp.
        upgraded = bucket[0]
        assert upgraded["candidate_ids"] == []
        assert upgraded["created_at"] is None

    def test_read_renders_validated_submitted_scored(self):
        """Full happy path: hypothesis → validate → submit → score
        merged in. read_scratchpad must render the chain.
        """
        from agents.openai_sdk.tools import build_handlers
        from agents.openai_sdk.core.history import (
            merge_results_into_state,
        )
        handlers = build_handlers(_make_challenge(with_flops=False))
        # validate creates a candidate with status="validated"
        validate_out = handlers["validate_code"](code=VALID_CODE)
        cid = [
            l for l in validate_out.splitlines()
            if l.startswith("candidate_id: ")
        ][-1].split(": ", 1)[1]
        # link hypothesis → candidate
        handlers["write_scratchpad"](
            hypothesis="larger patch sizes", candidate_id=cid,
        )
        # ship and merge a score for it
        self._ship(handlers, "", "m", "ship", candidate_id=cid)
        state = self._state(handlers)
        sub = state["submissions"][0]
        merge_results_into_state(state, [{
            "round_id": "r1", "code_hash": sub["code_hash"],
            "score": 0.81, "rank": 3, "rank_total": 9,
        }])
        out = handlers["read_scratchpad"]()
        assert "larger patch sizes" in out
        line = [l for l in out.splitlines() if cid in l][0]
        assert "validated" in line
        assert "submitted" in line
        assert "scored 0.81" in line
        assert "rank 3/9" in line

    def test_read_renders_pending_when_candidate_unknown(self):
        """A candidate_id with no matching candidate or submission
        record should still render — just say nothing's known yet."""
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge())
        handlers["write_scratchpad"](
            hypothesis="speculative idea",
            candidate_id="cand_deadbeef",
        )
        out = handlers["read_scratchpad"]()
        line = [l for l in out.splitlines() if "cand_deadbeef" in l][0]
        assert "no candidate state yet" in line

    def test_read_renders_validated_only(self):
        """Validated but not yet submitted: only the 'validated' tag
        appears, no 'submitted', no score."""
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge(with_flops=False))
        validate_out = handlers["validate_code"](code=VALID_CODE)
        cid = [
            l for l in validate_out.splitlines()
            if l.startswith("candidate_id: ")
        ][-1].split(": ", 1)[1]
        handlers["write_scratchpad"](
            hypothesis="just an idea", candidate_id=cid,
        )
        out = handlers["read_scratchpad"]()
        line = [l for l in out.splitlines() if cid in l][0]
        assert "validated" in line
        assert "submitted" not in line
        assert "scored" not in line

    def test_summary_line_counts_state(self):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge(with_flops=False))
        handlers["write_scratchpad"](hypothesis="h1")
        handlers["write_scratchpad"](hypothesis="h2")
        handlers["sketch_architecture"](code=(
            "import torch.nn as nn\n"
            "def build_model(context_len, prediction_len, num_variates,"
            " quantiles):\n"
            "    return nn.Linear(context_len, prediction_len)\n"
        ))
        self._ship(handlers, VALID_CODE, "m", "ship")
        out = handlers["read_scratchpad"]()
        summary = out.splitlines()[0]
        assert "2 hypotheses" in summary
        assert "1 candidates generated" in summary
        assert "1 submitted" in summary
        assert "no scores yet" in summary

    def test_summary_top_score_minimize(self):
        from agents.openai_sdk.tools import build_handlers
        from agents.openai_sdk.core.history import (
            merge_results_into_state,
        )
        ch = _make_challenge()
        ch["score_direction"] = "minimize"
        handlers = build_handlers(ch)
        handlers["write_scratchpad"](observation="ok")
        self._ship(handlers, VALID_CODE, "v1", "first")
        self._ship(
            handlers,
            VALID_CODE.replace("Adam", "Adam  # 2"),
            "v2", "second",
        )
        state = self._state(handlers)
        merge_results_into_state(state, [
            {"code_hash": state["submissions"][0]["code_hash"],
             "score": 0.5, "rank": 1, "round_id": "r1"},
            {"code_hash": state["submissions"][1]["code_hash"],
             "score": 0.2, "rank": 1, "round_id": "r2"},
        ])
        out = handlers["read_scratchpad"]()
        # Lower is better when minimizing → 0.2 wins.
        assert "top score: 0.2" in out.splitlines()[0]

    def test_summary_top_score_maximize(self):
        from agents.openai_sdk.tools import build_handlers
        from agents.openai_sdk.core.history import (
            merge_results_into_state,
        )
        ch = _make_challenge()
        ch["score_direction"] = "maximize"
        handlers = build_handlers(ch)
        handlers["write_scratchpad"](observation="ok")
        self._ship(handlers, VALID_CODE, "v1", "first")
        self._ship(
            handlers,
            VALID_CODE.replace("Adam", "Adam  # 2"),
            "v2", "second",
        )
        state = self._state(handlers)
        merge_results_into_state(state, [
            {"code_hash": state["submissions"][0]["code_hash"],
             "score": 0.5, "rank": 1, "round_id": "r1"},
            {"code_hash": state["submissions"][1]["code_hash"],
             "score": 0.2, "rank": 2, "round_id": "r2"},
        ])
        out = handlers["read_scratchpad"]()
        # Higher is better when maximizing → 0.5 wins.
        assert "top score: 0.5" in out.splitlines()[0]

    def test_empty_state_still_says_first_round(self):
        """The summary line alone shouldn't suppress the 'first round'
        message — read_scratchpad should treat an otherwise-empty
        state as not-yet-populated."""
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(_make_challenge())
        out = handlers["read_scratchpad"]()
        assert "first round" in out


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


def _build_wiki_tarball(entries: dict[str, str]) -> bytes:
    """Pack ``{name: content}`` into a gzipped tarball, return the bytes."""
    import io as _io
    import tarfile as _tarfile
    buf = _io.BytesIO()
    with _tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in entries.items():
            data = content.encode()
            info = _tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, _io.BytesIO(data))
    return buf.getvalue()


class TestCognitionWiki:
    """Wiki tools sit next to search_papers/query_db. When
    ``cognition_wiki_url`` is empty, both tools fail soft. When a
    tarball is published, the index is fetched once per round and
    cached in scratch_dir."""

    def test_index_when_url_empty_returns_not_published(self, tmp_path):
        from agents.openai_sdk.tools import build_handlers
        client = _MockClient()
        handlers = build_handlers(
            _make_challenge(cognition_wiki_url=""),
            client=client,
            scratch_dir=str(tmp_path),
        )
        out = handlers["cognition_wiki_index"]()
        assert "not published" in out.lower()
        assert client.get_call_count == 0

    def test_read_when_url_empty_returns_not_published(self, tmp_path):
        from agents.openai_sdk.tools import build_handlers
        client = _MockClient()
        handlers = build_handlers(
            _make_challenge(cognition_wiki_url=""),
            client=client,
            scratch_dir=str(tmp_path),
        )
        out = handlers["cognition_wiki_read"](slug="patchtst")
        assert "not published" in out.lower()
        assert client.get_call_count == 0

    def test_index_extracts_and_caches_tarball(self, tmp_path):
        from agents.openai_sdk.tools import build_handlers
        tar_bytes = _build_wiki_tarball({
            "_index.md": "# Test Wiki\n\n- foo: a foo entry\n",
            "foo.md": "# foo\n\nFull content of foo entry.\n",
        })
        client = _MockClient(get_response=tar_bytes)
        handlers = build_handlers(
            _make_challenge(
                cognition_wiki_url="https://r2.example/wiki.tar.gz",
            ),
            client=client,
            scratch_dir=str(tmp_path),
        )

        out = handlers["cognition_wiki_index"]()
        assert "Test Wiki" in out
        assert "foo" in out

        # Second call hits the on-disk cache, doesn't re-fetch.
        out2 = handlers["cognition_wiki_index"]()
        assert out2 == out
        assert client.get_call_count == 1

    def test_read_returns_entry_content(self, tmp_path):
        from agents.openai_sdk.tools import build_handlers
        tar_bytes = _build_wiki_tarball({
            "_index.md": "# index\n",
            "patchtst.md": "# patchtst\n\nClaim: patches help.\n",
        })
        handlers = build_handlers(
            _make_challenge(
                cognition_wiki_url="https://r2.example/wiki.tar.gz",
            ),
            client=_MockClient(get_response=tar_bytes),
            scratch_dir=str(tmp_path),
        )
        out = handlers["cognition_wiki_read"](slug="patchtst")
        assert "Claim: patches help." in out

    def test_read_unknown_slug_returns_not_found(self, tmp_path):
        from agents.openai_sdk.tools import build_handlers
        tar_bytes = _build_wiki_tarball({
            "_index.md": "# index\n",
            "foo.md": "# foo\n",
        })
        handlers = build_handlers(
            _make_challenge(
                cognition_wiki_url="https://r2.example/wiki.tar.gz",
            ),
            client=_MockClient(get_response=tar_bytes),
            scratch_dir=str(tmp_path),
        )
        out = handlers["cognition_wiki_read"](slug="missing")
        assert "not found" in out.lower()

    def test_read_rejects_path_traversal_and_bad_slugs(self, tmp_path):
        from agents.openai_sdk.tools import build_handlers
        handlers = build_handlers(
            _make_challenge(
                cognition_wiki_url="https://r2.example/wiki.tar.gz",
            ),
            client=_MockClient(),
            scratch_dir=str(tmp_path),
        )
        # Each bad slug must be rejected before any disk read. We accept
        # the circuit-breaker message too — it only fires on identical
        # error strings, which means the rejection is consistent.
        for bad in ("../etc/passwd", "foo/bar", "foo bar", "foo.md", ""):
            out = handlers["cognition_wiki_read"](slug=bad)
            low = out.lower()
            assert (
                out.startswith("error:")
                or "not found" in low
                or "unavailable" in low
                or "circuit open" in low
            ), f"expected reject for {bad!r}, got {out!r}"

    def test_index_handles_missing_index_file(self, tmp_path):
        """Tarball without _index.md → 'unavailable' string, no crash."""
        from agents.openai_sdk.tools import build_handlers
        tar_bytes = _build_wiki_tarball({"foo.md": "# foo\n"})
        handlers = build_handlers(
            _make_challenge(
                cognition_wiki_url="https://r2.example/wiki.tar.gz",
            ),
            client=_MockClient(get_response=tar_bytes),
            scratch_dir=str(tmp_path),
        )
        out = handlers["cognition_wiki_index"]()
        assert "unavailable" in out.lower()


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
        # Hypotheses are now structured dicts with candidate_ids.
        bucket = state["notes"]["open_hypotheses"]
        assert len(bucket) == 1
        assert bucket[0]["text"] == "try SSM backbone"
        assert bucket[0]["candidate_ids"] == []

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
        assert state["notes"]["open_hypotheses"][0]["text"] == "try attention"
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
        assert bucket[-1]["text"] == f"h{NOTES_MAX_ENTRIES + 2}"

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
        ch["agent_seconds"] = 180
        result = agent.design_architecture(ch, gated_client=None)
        assert result["code"] == VALID_CODE
        assert result["name"] == "instant"

    def test_returns_empty_failure_when_no_llm_code(self, mock_openai):
        """When the LLM never produces code AND there's no validated
        code stashed, the agent returns an honest-failure package
        (empty code, ``failed_<bucket>`` name) instead of masking the
        problem with a near-zero MLP fallback."""
        from agents.openai_sdk import agent

        # Always return plain text — no tool calls, no code blocks
        completion = _make_completion(content="thinking...")
        mock_openai.return_value.chat.completions.create.return_value = (
            completion
        )

        ch = _make_challenge()
        ch["agent_seconds"] = 180
        result = agent.design_architecture(ch, gated_client=None)
        assert result["name"].startswith("failed_")
        assert result["code"] == ""
        assert result["motivation"].startswith("FAILURE:")

    def test_chat_failure_returns_empty_failure(self, mock_openai):
        """If every LLM call raises and no validated code exists, the
        agent returns the honest-failure package."""
        from openai import APIError
        from agents.openai_sdk import agent
        err = APIError("bad", request=MagicMock(), body=None)
        err.status_code = 400  # non-transient → no retries inside chat()
        mock_openai.return_value.chat.completions.create.side_effect = err

        ch = _make_challenge()
        ch["agent_seconds"] = 180
        result = agent.design_architecture(ch, gated_client=None)
        assert result["name"].startswith("failed_")
        assert result["code"] == ""
        # Honest failure motivation surfaces the chat error.
        assert result["motivation"].startswith("FAILURE:")
        assert "LLM chat failed" in result["motivation"]

    def test_missing_llm_url_shows_config_error_motivation(
        self, mock_openai, monkeypatch
    ):
        """With no challenge['llm_url'] and no LLM_URL env var, the
        startup check should trip, skip the main loop, and return an
        honest-failure package whose motivation surfaces the config
        error (not the old generic 'could not produce code' message)."""
        _reset_llm_client_cache()
        monkeypatch.delenv("LLM_URL", raising=False)
        from agents.openai_sdk import agent

        ch = _make_challenge()
        ch["agent_seconds"] = 180
        # No llm_url key in challenge
        result = agent.design_architecture(ch, gated_client=None)

        assert result["name"].startswith("failed_")
        assert result["code"] == ""
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
        ch["agent_seconds"] = 180
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
