"""Tests for the autonomous agent — tool-calling loop, graceful degradation,
tool handlers, submit signal, time budget, and validation integration."""

import importlib.util
import json
import os
import sys
import tempfile
import time

# ── Module loading ────────────────────────────────────────────────
_agent_dir = os.path.join(os.path.dirname(__file__), "..", "agents", "autonomous")
sys.path.insert(0, _agent_dir)

# Clear cached core modules so we pick up autonomous agent's core
for _k in list(sys.modules.keys()):
    if _k == "core" or _k.startswith("core."):
        del sys.modules[_k]
    if _k == "tools" or _k.startswith("tools."):
        del sys.modules[_k]

_spec = importlib.util.spec_from_file_location(
    "autonomous_agent", os.path.join(_agent_dir, "agent.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

design_architecture = _mod.design_architecture
_build_system_prompt = _mod._build_system_prompt
_build_kickoff_message = _mod._build_kickoff_message
_autonomous_loop = _mod._autonomous_loop
MAX_TURNS = _mod.MAX_TURNS

from core.validation import validate_code
from core.history import (
    identify_bucket, extract_flops_budget, load_state, save_state,
    add_entry, SIZE_BUCKETS,
)
from tools import TOOLS, SubmitSignal, build_handlers


# ── Test helpers ──────────────────────────────────────────────────

class MockClient:
    """Mock GatedClient for testing."""
    def __init__(self, responses=None, raise_on_call=False):
        self.responses = responses or {}
        self.raise_on_call = raise_on_call
        self.calls = []

    def get_json(self, url):
        self.calls.append(("GET", url))
        if self.raise_on_call:
            raise RuntimeError("mock error")
        # Match by prefix for flexible URL matching
        for key, val in self.responses.items():
            if url.startswith(key) or url == key:
                return val
        return {}

    def post_json(self, url, payload):
        self.calls.append(("POST", url, payload))
        if self.raise_on_call:
            raise RuntimeError("mock error")
        for key, val in self.responses.items():
            if url.startswith(key) or url == key:
                if callable(val):
                    return val(payload)
                return val
        return {}


def _make_challenge(bucket="small", llm_url="", db_url="",
                    desearch_url="", frontier=None, time_budget=300):
    """Create a minimal challenge dict."""
    flops_min, flops_max = SIZE_BUCKETS.get(bucket, (500_000, 2_000_000))
    return {
        "min_flops_equivalent": flops_min,
        "max_flops_equivalent": flops_max,
        "feasible_frontier": frontier or [],
        "task": {
            "name": "ts_forecasting",
            "description": "Time series forecasting task",
            "task_params": {
                "context_len": 512,
                "prediction_len": 96,
                "num_variates": 1,
                "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            },
            "constraints": ["Output shape must be (batch, prediction_len, num_variates, len(quantiles))"],
            "objectives": [{"name": "crps", "primary": True}],
            "time_budget": time_budget,
        },
        "db_url": db_url,
        "desearch_url": desearch_url,
        "llm_url": llm_url,
        "seed": 42,
        "round_id": 1,
    }


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


def _make_llm_response(content="", tool_calls=None, finish_reason="stop"):
    """Build a mock LLM chat completion response."""
    msg = {"content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "choices": [{
            "message": msg,
            "finish_reason": finish_reason if not tool_calls else "tool_calls",
        }]
    }


def _make_tool_call(name, arguments, call_id="call_1"):
    """Build a mock tool call."""
    return {
        "id": call_id,
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


# ══════════════════════════════════════════════════════════════════
#  1. Tool definitions
# ══════════════════════════════════════════════════════════════════

class TestToolDefinitions:
    def test_tools_is_list(self):
        assert isinstance(TOOLS, list)

    def test_all_tools_have_required_fields(self):
        for tool in TOOLS:
            assert tool["type"] == "function"
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func

    def test_expected_tools_present(self):
        tool_names = {t["function"]["name"] for t in TOOLS}
        expected = {
            "search_papers", "query_db", "get_frontier_details",
            "estimate_model_flops", "validate_code", "read_scratchpad",
            "write_scratchpad", "submit", "time_remaining",
        }
        assert expected == tool_names

    def test_submit_requires_code_name_motivation(self):
        submit_tool = next(t for t in TOOLS if t["function"]["name"] == "submit")
        required = submit_tool["function"]["parameters"]["required"]
        assert "code" in required
        assert "name" in required
        assert "motivation" in required


# ══════════════════════════════════════════════════════════════════
#  2. Tool handlers
# ══════════════════════════════════════════════════════════════════

class TestToolHandlers:
    def _make_handlers(self, challenge=None, client=None, scratch_dir=None):
        challenge = challenge or _make_challenge()
        client = client or MockClient()
        scratch_dir = scratch_dir or tempfile.mkdtemp()
        deadline = time.time() + 300
        return build_handlers(client, challenge, scratch_dir, deadline)

    def test_all_handlers_present(self):
        handlers = self._make_handlers()
        expected = {
            "search_papers", "query_db", "get_frontier_details",
            "estimate_model_flops", "validate_code", "read_scratchpad",
            "write_scratchpad", "submit", "time_remaining",
        }
        assert set(handlers.keys()) == expected

    def test_time_remaining_returns_string(self):
        handlers = self._make_handlers()
        result = handlers["time_remaining"]()
        assert "remaining" in result.lower() or "seconds" in result.lower()

    def test_validate_code_passes_valid(self):
        # Use challenge without FLOPs to test structural validation only
        challenge = _make_challenge()
        challenge.pop("min_flops_equivalent", None)
        challenge.pop("max_flops_equivalent", None)
        challenge.pop("flops_budget", None)
        handlers = self._make_handlers(challenge=challenge)
        result = handlers["validate_code"](code=VALID_CODE)
        assert result.startswith("PASSED")

    def test_validate_code_fails_invalid(self):
        handlers = self._make_handlers()
        result = handlers["validate_code"](code="x = 1")
        assert result.startswith("FAILED")

    def test_estimate_flops_returns_info(self):
        handlers = self._make_handlers()
        result = handlers["estimate_model_flops"](code=VALID_CODE)
        assert "Estimated FLOPs" in result
        assert "Budget" in result

    def test_frontier_details_empty(self):
        handlers = self._make_handlers()
        result = handlers["get_frontier_details"]()
        assert "bootstrapping" in result.lower() or "no frontier" in result.lower()

    def test_frontier_details_with_data(self):
        frontier = [{"objectives": {"crps": 0.4}, "code": "x=1"}]
        challenge = _make_challenge(frontier=frontier)
        handlers = self._make_handlers(challenge=challenge)
        result = handlers["get_frontier_details"]()
        assert "0.4" in result

    def test_scratchpad_empty_first_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            handlers = self._make_handlers(scratch_dir=tmp)
            result = handlers["read_scratchpad"]()
            assert "empty" in result.lower() or "first round" in result.lower()

    def test_scratchpad_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            handlers = self._make_handlers(scratch_dir=tmp)
            handlers["write_scratchpad"](notes="LinearMLP worked well in small bucket")
            result = handlers["read_scratchpad"]()
            assert "LinearMLP" in result

    def test_search_papers_no_url(self):
        challenge = _make_challenge(desearch_url="")
        handlers = self._make_handlers(challenge=challenge)
        result = handlers["search_papers"](query="transformer forecasting")
        assert "unavailable" in result.lower()

    def test_query_db_no_url(self):
        challenge = _make_challenge(db_url="")
        handlers = self._make_handlers(challenge=challenge)
        result = handlers["query_db"](path="/experiments/recent")
        assert "unavailable" in result.lower()

    def test_query_db_get(self):
        """query_db GET returns formatted JSON from the DB."""
        client = MockClient(responses={
            "http://db/experiments/recent": {"experiments": [{"name": "exp1", "crps": 0.42}]},
        })
        challenge = _make_challenge(db_url="http://db")
        handlers = self._make_handlers(challenge=challenge, client=client)
        result = handlers["query_db"](path="/experiments/recent")
        assert "exp1" in result
        assert "0.42" in result

    def test_query_db_get_with_query_params(self):
        """query_db passes query params through in the path."""
        client = MockClient(responses={
            "http://db/experiments/recent?n=5": {"experiments": [{"name": "exp2"}]},
        })
        challenge = _make_challenge(db_url="http://db")
        handlers = self._make_handlers(challenge=challenge, client=client)
        result = handlers["query_db"](path="/experiments/recent?n=5")
        assert "exp2" in result

    def test_query_db_post(self):
        """query_db POST sends a JSON body and returns results."""
        client = MockClient(responses={
            "http://db/experiments/search": {"results": [{"name": "found_it"}]},
        })
        challenge = _make_challenge(db_url="http://db")
        handlers = self._make_handlers(challenge=challenge, client=client)
        result = handlers["query_db"](
            path="/experiments/search",
            method="POST",
            body={"bucket": "small", "min_crps": 0.3},
        )
        assert "found_it" in result

    def test_query_db_error_handling(self):
        """query_db returns error message on failure, doesn't crash."""
        client = MockClient(raise_on_call=True)
        challenge = _make_challenge(db_url="http://db")
        handlers = self._make_handlers(challenge=challenge, client=client)
        result = handlers["query_db"](path="/experiments/recent")
        assert "failed" in result.lower()

    def test_query_db_error_response(self):
        """query_db reports when DB returns an error field."""
        client = MockClient(responses={
            "http://db/bad/path": {"error": "not found"},
        })
        challenge = _make_challenge(db_url="http://db")
        handlers = self._make_handlers(challenge=challenge, client=client)
        result = handlers["query_db"](path="/bad/path")
        assert "not found" in result

    def test_query_db_truncates_large_response(self):
        """Very large DB responses get truncated to avoid flooding context."""
        huge = {"data": "x" * 10000}
        client = MockClient(responses={
            "http://db/huge": huge,
        })
        challenge = _make_challenge(db_url="http://db")
        handlers = self._make_handlers(challenge=challenge, client=client)
        result = handlers["query_db"](path="/huge")
        assert len(result) <= 8100  # 8000 + truncation message
        assert "truncated" in result

    def test_query_db_prepends_slash(self):
        """Path without leading slash still works."""
        client = MockClient(responses={
            "http://db/experiments/recent": {"ok": True},
        })
        challenge = _make_challenge(db_url="http://db")
        handlers = self._make_handlers(challenge=challenge, client=client)
        result = handlers["query_db"](path="experiments/recent")
        assert "ok" in result

    def test_submit_valid_raises_signal(self):
        # Use challenge without FLOPs for structural-only validation
        challenge = _make_challenge()
        challenge.pop("min_flops_equivalent", None)
        challenge.pop("max_flops_equivalent", None)
        handlers = self._make_handlers(challenge=challenge)
        try:
            handlers["submit"](
                code=VALID_CODE,
                name="test_model",
                motivation="testing submit",
            )
            assert False, "Should have raised SubmitSignal"
        except SubmitSignal as sig:
            assert sig.code == VALID_CODE
            assert sig.name == "test_model"
            assert sig.motivation == "testing submit"

    def test_submit_invalid_returns_error(self):
        handlers = self._make_handlers()
        result = handlers["submit"](
            code="x = 1",
            name="bad_model",
            motivation="should fail",
        )
        assert "REJECTED" in result
        assert "validation" in result.lower()


# ══════════════════════════════════════════════════════════════════
#  3. System prompt
# ══════════════════════════════════════════════════════════════════

class TestSystemPrompt:
    def test_contains_task_params(self):
        challenge = _make_challenge()
        prompt = _build_system_prompt(challenge)
        assert "context_len" in prompt
        assert "prediction_len" in prompt

    def test_contains_budget(self):
        challenge = _make_challenge("medium")
        prompt = _build_system_prompt(challenge)
        assert "medium" in prompt.lower()
        assert "FLOPs" in prompt

    def test_contains_workflow_guidance(self):
        challenge = _make_challenge()
        prompt = _build_system_prompt(challenge)
        assert "validate" in prompt.lower()
        assert "submit" in prompt.lower()
        assert "scratchpad" in prompt.lower()

    def test_contains_code_requirements(self):
        challenge = _make_challenge()
        prompt = _build_system_prompt(challenge)
        assert "build_model" in prompt
        assert "build_optimizer" in prompt
        assert "subprocess" in prompt

    def test_contains_domain_context(self):
        challenge = _make_challenge()
        challenge["task"]["domain_system_prompt"] = "You are a time series expert."
        prompt = _build_system_prompt(challenge)
        assert "time series expert" in prompt

    def test_contains_constraints(self):
        challenge = _make_challenge()
        prompt = _build_system_prompt(challenge)
        assert "Output shape" in prompt


# ══════════════════════════════════════════════════════════════════
#  4. Kickoff message
# ══════════════════════════════════════════════════════════════════

class TestKickoffMessage:
    def test_contains_task_info(self):
        challenge = _make_challenge()
        msg = _build_kickoff_message(challenge)
        assert "ts_forecasting" in msg
        assert "small" in msg

    def test_contains_frontier_count(self):
        frontier = [{"objectives": {"crps": 0.4}, "code": "x=1"}]
        challenge = _make_challenge(frontier=frontier)
        msg = _build_kickoff_message(challenge)
        assert "1 model" in msg

    def test_contains_time_budget(self):
        challenge = _make_challenge(time_budget=120)
        msg = _build_kickoff_message(challenge)
        assert "120" in msg


# ══════════════════════════════════════════════════════════════════
#  5. Autonomous loop
# ══════════════════════════════════════════════════════════════════

class TestAutonomousLoop:
    def test_submit_on_first_turn(self):
        """Agent calls submit on the first turn — result is captured."""
        challenge = _make_challenge(llm_url="http://llm")
        challenge.pop("min_flops_equivalent", None)
        challenge.pop("max_flops_equivalent", None)

        # LLM responds with a submit tool call
        tool_call = _make_tool_call("submit", {
            "code": VALID_CODE,
            "name": "instant_model",
            "motivation": "just submit it",
        })
        client = MockClient(responses={
            "http://llm/v1/chat/completions": _make_llm_response(
                tool_calls=[tool_call]
            ),
        })

        handlers = build_handlers(
            client, challenge, tempfile.mkdtemp(), time.time() + 300)
        messages = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "go"},
        ]
        result = _autonomous_loop(client, challenge, messages, handlers, time.time() + 300)
        assert result is not None
        assert result["code"] == VALID_CODE
        assert result["name"] == "instant_model"

    def test_tool_calls_honored_when_finish_reason_stop(self):
        """Regression: some Kimi/OpenAI-compatible servers return
        ``finish_reason="stop"`` together with a populated ``tool_calls``
        list. The loop must still execute those tool calls instead of
        treating the turn as a plain text response."""
        challenge = _make_challenge(llm_url="http://llm")
        challenge.pop("min_flops_equivalent", None)
        challenge.pop("max_flops_equivalent", None)

        tool_call = _make_tool_call("submit", {
            "code": VALID_CODE,
            "name": "kimi_stop_model",
            "motivation": "finish_reason=stop shouldn't drop tool calls",
        })
        # Build the response manually so we can force finish_reason="stop"
        # *alongside* tool_calls — the combination _make_llm_response avoids.
        raw_response = {
            "choices": [{
                "message": {"content": "", "tool_calls": [tool_call]},
                "finish_reason": "stop",
            }]
        }
        client = MockClient(responses={
            "http://llm/v1/chat/completions": raw_response,
        })

        handlers = build_handlers(
            client, challenge, tempfile.mkdtemp(), time.time() + 300)
        messages = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "go"},
        ]
        result = _autonomous_loop(
            client, challenge, messages, handlers, time.time() + 300)
        assert result is not None, "tool_calls were dropped on finish_reason=stop"
        assert result["code"] == VALID_CODE
        assert result["name"] == "kimi_stop_model"

    def test_validate_then_submit(self):
        """Agent validates first, then submits — two-turn interaction."""
        challenge = _make_challenge(llm_url="http://llm")
        challenge.pop("min_flops_equivalent", None)
        challenge.pop("max_flops_equivalent", None)

        call_count = [0]

        def mock_llm(payload):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: validate
                return _make_llm_response(tool_calls=[
                    _make_tool_call("validate_code", {"code": VALID_CODE}, "call_v1"),
                ])
            else:
                # Second call: submit
                return _make_llm_response(tool_calls=[
                    _make_tool_call("submit", {
                        "code": VALID_CODE,
                        "name": "validated_model",
                        "motivation": "passed validation",
                    }, "call_s1"),
                ])

        client = MockClient(responses={
            "http://llm/v1/chat/completions": mock_llm,
        })

        handlers = build_handlers(
            client, challenge, tempfile.mkdtemp(), time.time() + 300)
        messages = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "go"},
        ]
        result = _autonomous_loop(client, challenge, messages, handlers, time.time() + 300)
        assert result is not None
        assert result["name"] == "validated_model"
        assert call_count[0] == 2

    def test_no_llm_url_returns_none(self):
        """No LLM URL — loop returns None immediately."""
        challenge = _make_challenge(llm_url="")
        handlers = {}
        messages = []
        result = _autonomous_loop(
            MockClient(), challenge, messages, handlers, time.time() + 300)
        assert result is None

    def test_time_expired_returns_none(self):
        """Deadline already passed — loop returns None."""
        challenge = _make_challenge(llm_url="http://llm")
        handlers = {}
        messages = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "go"},
        ]
        result = _autonomous_loop(
            MockClient(), challenge, messages, handlers, time.time() - 1)
        assert result is None

    def test_fallback_to_last_validated(self):
        """If turns exhaust without submit, use last validated code."""
        challenge = _make_challenge(llm_url="http://llm")
        challenge.pop("min_flops_equivalent", None)
        challenge.pop("max_flops_equivalent", None)

        call_count = [0]

        def mock_llm(payload):
            call_count[0] += 1
            if call_count[0] == 1:
                # Validate code
                return _make_llm_response(tool_calls=[
                    _make_tool_call("validate_code", {"code": VALID_CODE}, "call_v1"),
                ])
            else:
                # Keep returning text (never submits)
                return _make_llm_response(content="Hmm, let me think more...")

        client = MockClient(responses={
            "http://llm/v1/chat/completions": mock_llm,
        })

        handlers = build_handlers(
            client, challenge, tempfile.mkdtemp(), time.time() + 300)
        messages = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "go"},
        ]

        # Patch MAX_TURNS to limit test time
        original_max = _mod.MAX_TURNS
        _mod.MAX_TURNS = 4
        try:
            result = _autonomous_loop(
                client, challenge, messages, handlers, time.time() + 300)
        finally:
            _mod.MAX_TURNS = original_max

        assert result is not None
        assert result["name"] == "autonomous_fallback"
        assert result["code"] == VALID_CODE

    def test_fallback_to_proposed_when_flops_off_gate(self):
        """If validate_code fails ONLY on FLOPs, the structurally-ok code
        should still be captured as ``last_proposed_code`` and returned as
        ``autonomous_best_effort`` when the loop exhausts. Previously the
        loop returned None and the agent shipped literal empty code, which
        the harness rejected with ``Missing build_model``."""
        # Use impossibly-high FLOPs bounds so VALID_CODE (a trivial linear
        # model) definitely fails the FLOPs hard gate but still passes
        # structural checks (has build_model + build_optimizer at top level).
        challenge = _make_challenge(llm_url="http://llm")
        challenge["min_flops_equivalent"] = 10 ** 15
        challenge["max_flops_equivalent"] = 10 ** 16

        def mock_llm(payload):
            # Every turn the LLM calls validate_code with the same code.
            return _make_llm_response(tool_calls=[
                _make_tool_call(
                    "validate_code", {"code": VALID_CODE}, "call_v1"),
            ])

        client = MockClient(responses={
            "http://llm/v1/chat/completions": mock_llm,
        })
        handlers = build_handlers(
            client, challenge, tempfile.mkdtemp(), time.time() + 300)
        messages = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "go"},
        ]

        original_max = _mod.MAX_TURNS
        _mod.MAX_TURNS = 3
        try:
            result = _autonomous_loop(
                client, challenge, messages, handlers, time.time() + 300)
        finally:
            _mod.MAX_TURNS = original_max

        assert result is not None, (
            "loop returned None — off-gate proposals should fall back to "
            "last_proposed_code instead of nothing"
        )
        assert result["name"] == "autonomous_best_effort"
        assert result["code"] == VALID_CODE

    def test_llm_failure_all_retries(self):
        """All LLM retries fail — loop returns None."""
        challenge = _make_challenge(llm_url="http://llm")
        client = MockClient(raise_on_call=True)

        handlers = build_handlers(
            client, challenge, tempfile.mkdtemp(), time.time() + 300)
        messages = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "go"},
        ]
        result = _autonomous_loop(
            client, challenge, messages, handlers, time.time() + 300)
        assert result is None

    def test_multiple_tool_calls_in_one_turn(self):
        """Agent calls multiple tools in one turn — all are processed."""
        challenge = _make_challenge(llm_url="http://llm")
        challenge.pop("min_flops_equivalent", None)
        challenge.pop("max_flops_equivalent", None)

        call_count = [0]

        def mock_llm(payload):
            call_count[0] += 1
            if call_count[0] == 1:
                # Multiple tool calls: time_remaining + validate
                return _make_llm_response(tool_calls=[
                    _make_tool_call("time_remaining", {}, "call_t1"),
                    _make_tool_call("validate_code", {"code": VALID_CODE}, "call_v1"),
                ])
            else:
                return _make_llm_response(tool_calls=[
                    _make_tool_call("submit", {
                        "code": VALID_CODE,
                        "name": "multi_tool",
                        "motivation": "used multiple tools",
                    }, "call_s1"),
                ])

        client = MockClient(responses={
            "http://llm/v1/chat/completions": mock_llm,
        })

        handlers = build_handlers(
            client, challenge, tempfile.mkdtemp(), time.time() + 300)
        messages = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "go"},
        ]
        result = _autonomous_loop(
            client, challenge, messages, handlers, time.time() + 300)
        assert result is not None
        assert result["name"] == "multi_tool"

    def test_unknown_tool_handled_gracefully(self):
        """Agent calls a tool that doesn't exist — gets error, continues."""
        challenge = _make_challenge(llm_url="http://llm")
        challenge.pop("min_flops_equivalent", None)
        challenge.pop("max_flops_equivalent", None)

        call_count = [0]

        def mock_llm(payload):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_llm_response(tool_calls=[
                    _make_tool_call("nonexistent_tool", {}, "call_x1"),
                ])
            else:
                return _make_llm_response(tool_calls=[
                    _make_tool_call("submit", {
                        "code": VALID_CODE,
                        "name": "recovered",
                        "motivation": "recovered from bad tool call",
                    }, "call_s1"),
                ])

        client = MockClient(responses={
            "http://llm/v1/chat/completions": mock_llm,
        })

        handlers = build_handlers(
            client, challenge, tempfile.mkdtemp(), time.time() + 300)
        messages = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "go"},
        ]
        result = _autonomous_loop(
            client, challenge, messages, handlers, time.time() + 300)
        assert result is not None
        assert result["name"] == "recovered"


# ══════════════════════════════════════════════════════════════════
#  6. Graceful degradation — design_architecture entry point
# ══════════════════════════════════════════════════════════════════

class TestGracefulDegradation:
    def _run(self, challenge, client):
        _mod.load_scratchpad = lambda c: None
        _mod.save_scratchpad = lambda c, d: None
        import builtins
        builtins.load_scratchpad = lambda c: None
        builtins.save_scratchpad = lambda c, d: None
        try:
            return design_architecture(challenge, client)
        finally:
            del builtins.load_scratchpad
            del builtins.save_scratchpad

    def test_no_llm_uses_fallback_template(self):
        """When no LLM URL is provided, fallback template provides valid code."""
        challenge = _make_challenge("small", llm_url="")
        client = MockClient()
        result = self._run(challenge, client)
        # Fallback template should produce structurally valid code
        assert result["code"] != ""
        assert "fallback" in result["name"]
        assert "build_model" in result["code"]
        assert "build_optimizer" in result["code"]

    def test_llm_failure_uses_fallback_template(self):
        """When the LLM is unreachable, fallback template provides valid code."""
        challenge = _make_challenge("small", llm_url="http://fake-llm")
        client = MockClient(raise_on_call=True)
        result = self._run(challenge, client)
        # Fallback template should produce structurally valid code
        assert result["code"] != ""
        assert "fallback" in result["name"]
        assert "build_model" in result["code"]

    def test_does_not_crash_without_llm(self):
        challenge = _make_challenge("medium", llm_url="")
        client = MockClient()
        result = self._run(challenge, client)
        assert isinstance(result, dict)
        assert "code" in result
        assert "name" in result
        assert "motivation" in result

    def test_every_bucket_no_crash(self):
        for bucket in SIZE_BUCKETS:
            challenge = _make_challenge(bucket, llm_url="")
            client = MockClient()
            result = self._run(challenge, client)
            assert isinstance(result, dict), f"Non-dict for {bucket}"
            assert "code" in result

    def test_successful_submission(self):
        """Full flow: LLM validates and submits on first turn."""
        challenge = _make_challenge("small", llm_url="http://llm")
        # Remove FLOPs bounds so VALID_CODE passes structural validation
        challenge.pop("min_flops_equivalent", None)
        challenge.pop("max_flops_equivalent", None)

        tool_call = _make_tool_call("submit", {
            "code": VALID_CODE,
            "name": "full_flow_model",
            "motivation": "end-to-end test",
        })
        client = MockClient(responses={
            "http://llm/v1/chat/completions": _make_llm_response(
                tool_calls=[tool_call]
            ),
        })

        result = self._run(challenge, client)
        assert result["code"] == VALID_CODE
        assert result["name"] == "full_flow_model"


# ══════════════════════════════════════════════════════════════════
#  7. SubmitSignal
# ══════════════════════════════════════════════════════════════════

class TestSubmitSignal:
    def test_carries_data(self):
        sig = SubmitSignal("code", "name", "motivation")
        assert sig.code == "code"
        assert sig.name == "name"
        assert sig.motivation == "motivation"

    def test_is_exception(self):
        sig = SubmitSignal("c", "n", "m")
        assert isinstance(sig, Exception)


# ══════════════════════════════════════════════════════════════════
#  8. MAX_TURNS constant
# ══════════════════════════════════════════════════════════════════

class TestConstants:
    def test_max_turns_is_25(self):
        """MAX_TURNS should be 25 (leaves 5 headroom from 30-request limit)."""
        assert MAX_TURNS == 25

    def test_time_buffer_is_positive(self):
        assert _mod.TIME_BUFFER_SECONDS > 0


# ══════════════════════════════════════════════════════════════════
#  9. Integration: validation works end-to-end
# ══════════════════════════════════════════════════════════════════

class TestValidationIntegration:
    def test_valid_code_passes(self):
        ok, errors = validate_code(VALID_CODE)
        assert ok, f"Valid code rejected: {errors}"

    def test_empty_code_rejected(self):
        ok, _ = validate_code("")
        assert not ok

    def test_missing_functions_rejected(self):
        ok, errors = validate_code("x = 1")
        assert not ok
        assert any("build_model" in e for e in errors)
