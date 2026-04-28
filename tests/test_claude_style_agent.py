"""Tests for the claude_style multi-subagent miner agent.

Mirrors the openai_sdk test setup: the OpenAI client is mocked at
module level — no real network calls. Module / sys.path bookkeeping
mirrors test_openai_sdk_agent so a single pytest run can load both
agents back-to-back without their cached imports conflicting.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest


# ── Module loading ────────────────────────────────────────────────
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _purge_shared_modules() -> None:
    """Drop ``core.*`` / ``tools`` / ``prompts`` / ``llm_client`` /
    ``subagents.*`` plus the claude_style submodules so each test
    re-resolves imports against the claude_style package and tests
    don't pick up state from a prior openai_sdk import.
    """
    for k in list(sys.modules.keys()):
        if k == "core" or k.startswith("core."):
            del sys.modules[k]
        if k in ("tools", "prompts", "llm_client", "hooks"):
            del sys.modules[k]
        if k == "subagents" or k.startswith("subagents."):
            del sys.modules[k]
        if k.startswith("agents.claude_style"):
            del sys.modules[k]


def _reset_llm_client_cache() -> None:
    from agents.claude_style import llm_client
    llm_client._cached_client = None
    llm_client._cached_config = None


@pytest.fixture(autouse=True)
def _isolate_claude_style_imports():
    """Purge shared module names before every test in this file.

    Other test modules (e.g. ``test_autonomous.py``) put their own
    agent's directory on ``sys.path`` and bind ``core.*`` /
    ``tools`` in ``sys.modules`` to that agent's submodules. Without
    this autouse purge, the FIRST test that imports
    ``agents.claude_style.tools`` would re-resolve ``from
    core.history import ...`` against whatever ``core.history``
    happens to still be cached — typically autonomous's, which
    lacks symbols claude_style depends on.
    """
    _purge_shared_modules()
    yield
    _purge_shared_modules()


@pytest.fixture
def mock_openai(monkeypatch):
    monkeypatch.setenv("LLM_URL", "http://test/llm")
    monkeypatch.setenv("AGENT_TOKEN", "test-token")
    monkeypatch.setenv("MINER_UID", "42")
    with patch("agents.claude_style.llm_client.OpenAI") as m:
        _reset_llm_client_cache()
        yield m
    _reset_llm_client_cache()


def _make_completion(content: str = "", tool_calls=None, finish_reason="stop"):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
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
    choice.finish_reason = finish_reason if not tool_calls else "tool_calls"
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


def _make_challenge(with_flops: bool = True) -> dict:
    ch = {
        "feasible_frontier": [],
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


# ══════════════════════════════════════════════════════════════════
#  1. tools.build_tools — role parameter splits the surface
# ══════════════════════════════════════════════════════════════════

class TestRoleToolSubsets:
    def test_researcher_role_returns_research_tools(self):
        from agents.claude_style.tools import build_tools
        names = {
            t["function"]["name"]
            for t in build_tools(_make_challenge(), role="researcher")
        }
        assert {
            "search_papers", "query_db", "list_frontier", "analyze_task",
        } <= names
        # Designer-only tools must NOT be in the researcher subset.
        assert "submit" not in names
        assert "validate_code" not in names
        assert "sketch_architecture" not in names

    def test_designer_role_returns_design_tools(self):
        from agents.claude_style.tools import build_tools
        names = {
            t["function"]["name"]
            for t in build_tools(_make_challenge(), role="designer")
        }
        assert {
            "sketch_architecture", "estimate_layer_flops",
            "validate_code", "submit",
        } <= names
        # Researcher-only tools must NOT be in the designer subset.
        assert "search_papers" not in names
        assert "query_db" not in names

    def test_unknown_role_raises(self):
        from agents.claude_style.tools import build_tools
        with pytest.raises(ValueError):
            build_tools(_make_challenge(), role="potato")

    def test_no_role_returns_full_surface(self):
        from agents.claude_style.tools import build_tools
        names = {
            t["function"]["name"]
            for t in build_tools(_make_challenge())
        }
        # Same baseline the openai_sdk test pins.
        assert "submit" in names
        assert "search_papers" in names
        assert "validate_code" in names

    def test_schemas_are_byte_identical_to_full_surface(self):
        """Splitting tools by role must NOT change the schema content
        of any tool — the parameter spec, descriptions, and required
        lists stay exactly as the full TOOLS list defines them."""
        from agents.claude_style.tools import TOOLS, build_tools
        full = {t["function"]["name"]: t for t in build_tools(_make_challenge())}
        researcher = build_tools(_make_challenge(), role="researcher")
        for t in researcher:
            name = t["function"]["name"]
            assert t == full[name], f"role-filtered {name} differs"


# ══════════════════════════════════════════════════════════════════
#  2. hooks — submit_requires_recent_validate
# ══════════════════════════════════════════════════════════════════

class TestSubmitHook:
    def test_blocks_when_no_validate_history(self):
        from agents.claude_style.hooks import (
            SUBMIT_BLOCKED_MSG, submit_requires_recent_validate,
        )
        assert (
            submit_requires_recent_validate("submit", {}, {})
            == SUBMIT_BLOCKED_MSG
        )

    def test_allows_when_recent_ok(self):
        from agents.claude_style.hooks import submit_requires_recent_validate
        state = {"validate_history": [{"round": 5, "ok": True}]}
        assert (
            submit_requires_recent_validate("submit", {}, state) is None
        )

    def test_blocks_when_recent_failures_only(self):
        from agents.claude_style.hooks import (
            SUBMIT_BLOCKED_MSG, submit_requires_recent_validate,
        )
        state = {"validate_history": [
            {"round": 3, "ok": False},
            {"round": 4, "ok": False},
            {"round": 5, "ok": False},
        ]}
        assert (
            submit_requires_recent_validate("submit", {}, state)
            == SUBMIT_BLOCKED_MSG
        )

    def test_blocks_when_old_ok_outside_lookback(self):
        """An ``ok=True`` from before the lookback window must not
        let a fresh submit through."""
        from agents.claude_style.hooks import (
            SUBMIT_BLOCKED_MSG, submit_requires_recent_validate,
        )
        # Latest round is 10, lookback is 3 → only rounds > 7 count.
        state = {"validate_history": [
            {"round": 6, "ok": True},
            {"round": 8, "ok": False},
            {"round": 10, "ok": False},
        ]}
        assert (
            submit_requires_recent_validate("submit", {}, state)
            == SUBMIT_BLOCKED_MSG
        )

    def test_lets_non_submit_through(self):
        from agents.claude_style.hooks import submit_requires_recent_validate
        # Hook is keyed on submit; everything else returns None.
        assert (
            submit_requires_recent_validate("validate_code", {}, {})
            is None
        )


# ══════════════════════════════════════════════════════════════════
#  3. researcher — JSON brief round-trip + retry + default fallback
# ══════════════════════════════════════════════════════════════════

GOOD_BRIEF = {
    "relevant_prior_work": ["PatchTST", "iTransformer"],
    "frontier_gaps": ["no models exploit channel mixing"],
    "ideas_to_try": ["depthwise-sep conv backbone"],
    "plan": [
        "sketch a small candidate",
        "size to mid-bucket",
        "validate and submit",
    ],
}


def _brief_completion(brief: dict) -> MagicMock:
    """Assistant turn with the brief fenced as ```json."""
    return _make_completion(
        content=f"```json\n{json.dumps(brief)}\n```",
    )


class TestResearcher:
    def test_brief_parses_from_fenced_json(self, mock_openai):
        from agents.claude_style.subagents.researcher import run_researcher
        from agents.claude_style.tools import build_handlers
        mock_openai.return_value.chat.completions.create.return_value = (
            _brief_completion(GOOD_BRIEF)
        )
        ch = _make_challenge()
        handlers = build_handlers(ch)
        import time as _time
        brief = run_researcher(
            challenge=ch,
            handlers=handlers,
            deadline=_time.monotonic() + 60,
            llm_kwargs={
                "llm_url": "http://test/llm",
                "agent_token": "t",
                "miner_uid": "0",
                "model": "test-model",
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            state={},
            bucket="small",
        )
        assert brief["plan"] == GOOD_BRIEF["plan"]
        assert brief["relevant_prior_work"] == GOOD_BRIEF["relevant_prior_work"]
        # Made one chat call — landed first try.
        assert (
            mock_openai.return_value.chat.completions.create.call_count
            == 1
        )

    def test_brief_extracts_from_bare_braces(self, mock_openai):
        """No fence — brace-balanced scan should still find the brief."""
        from agents.claude_style.subagents.researcher import run_researcher
        from agents.claude_style.tools import build_handlers
        mock_openai.return_value.chat.completions.create.return_value = (
            _make_completion(content="here you go: " + json.dumps(GOOD_BRIEF))
        )
        ch = _make_challenge()
        import time as _time
        brief = run_researcher(
            challenge=ch,
            handlers=build_handlers(ch),
            deadline=_time.monotonic() + 60,
            llm_kwargs={
                "llm_url": "http://test/llm",
                "agent_token": "t",
                "miner_uid": "0",
                "model": "test-model",
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            state={},
            bucket="small",
        )
        assert brief["plan"] == GOOD_BRIEF["plan"]

    def test_retries_once_then_succeeds(self, mock_openai):
        """First call returns prose, second call returns the brief."""
        from agents.claude_style.subagents.researcher import run_researcher
        from agents.claude_style.tools import build_handlers
        mock_openai.return_value.chat.completions.create.side_effect = [
            _make_completion(content="thinking..."),
            _brief_completion(GOOD_BRIEF),
        ]
        ch = _make_challenge()
        import time as _time
        brief = run_researcher(
            challenge=ch,
            handlers=build_handlers(ch),
            deadline=_time.monotonic() + 60,
            llm_kwargs={
                "llm_url": "http://test/llm",
                "agent_token": "t",
                "miner_uid": "0",
                "model": "test-model",
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            state={},
            bucket="small",
        )
        assert brief["plan"] == GOOD_BRIEF["plan"]
        # First attempt failed → retry kicked in → 2 calls total.
        assert (
            mock_openai.return_value.chat.completions.create.call_count
            == 2
        )

    def test_falls_back_to_default_when_both_attempts_fail(
        self, mock_openai,
    ):
        from agents.claude_style.subagents.researcher import run_researcher
        from agents.claude_style.tools import build_handlers
        # Every call returns prose with no JSON object.
        mock_openai.return_value.chat.completions.create.return_value = (
            _make_completion(content="I refuse to comply.")
        )
        ch = _make_challenge()
        import time as _time
        brief = run_researcher(
            challenge=ch,
            handlers=build_handlers(ch),
            deadline=_time.monotonic() + 60,
            llm_kwargs={
                "llm_url": "http://test/llm",
                "agent_token": "t",
                "miner_uid": "0",
                "model": "test-model",
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            state={},
            bucket="small",
        )
        # Default brief carries the tag.
        assert brief.get("_default") is True
        assert isinstance(brief["plan"], list)
        # First + retry → 2 calls.
        assert (
            mock_openai.return_value.chat.completions.create.call_count
            == 2
        )

    def test_normalizes_non_list_fields(self, mock_openai):
        """If the LLM returns a string where a list is expected, the
        normalizer wraps it so the designer can iterate safely."""
        from agents.claude_style.subagents.researcher import run_researcher
        from agents.claude_style.tools import build_handlers
        sloppy = {
            "relevant_prior_work": "PatchTST",  # string, not list
            "frontier_gaps": [],
            "ideas_to_try": [],
            "plan": "just submit",  # string, not list
        }
        mock_openai.return_value.chat.completions.create.return_value = (
            _brief_completion(sloppy)
        )
        ch = _make_challenge()
        import time as _time
        brief = run_researcher(
            challenge=ch,
            handlers=build_handlers(ch),
            deadline=_time.monotonic() + 60,
            llm_kwargs={
                "llm_url": "http://test/llm",
                "agent_token": "t",
                "miner_uid": "0",
                "model": "test-model",
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            state={},
            bucket="small",
        )
        assert brief["relevant_prior_work"] == ["PatchTST"]
        assert brief["plan"] == ["just submit"]


# ══════════════════════════════════════════════════════════════════
#  4. orchestrator — researcher step round-trips end-to-end
# ══════════════════════════════════════════════════════════════════

class TestOrchestratorResearcher:
    def test_design_architecture_runs_researcher_then_falls_back(
        self, mock_openai,
    ):
        """End-to-end smoke: orchestrator instantiates the researcher,
        the (stub) designer doesn't ship, and the orchestrator falls
        through to the fallback template — proves the researcher slot
        is wired correctly without depending on the designer being
        implemented yet."""
        from agents.claude_style import agent
        # Researcher returns a brief; designer is still a stub
        # (returns None), so the orchestrator falls back.
        mock_openai.return_value.chat.completions.create.return_value = (
            _brief_completion(GOOD_BRIEF)
        )
        ch = _make_challenge()
        ch["agent_seconds"] = 180
        result = agent.design_architecture(ch, gated_client=None)
        assert "code" in result and "name" in result and "motivation" in result
        # Designer stub returned None, so we expect a fallback template
        # name (or auto_submit_*).
        assert (
            result["name"].startswith("fallback_")
            or result["name"].startswith("auto_submit_")
        )
