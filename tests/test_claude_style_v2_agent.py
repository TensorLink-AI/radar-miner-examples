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
        if k.startswith("agents.claude_style_v2"):
            del sys.modules[k]


def _reset_llm_client_cache() -> None:
    from agents.claude_style_v2 import llm_client
    llm_client._cached_client = None
    llm_client._cached_config = None


@pytest.fixture(autouse=True)
def _isolate_claude_style_imports():
    """Purge shared module names before every test in this file.

    Other test modules (e.g. ``test_autonomous.py``) put their own
    agent's directory on ``sys.path`` and bind ``core.*`` /
    ``tools`` in ``sys.modules`` to that agent's submodules. Without
    this autouse purge, the FIRST test that imports
    ``agents.claude_style_v2.tools`` would re-resolve ``from
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
    with patch("agents.claude_style_v2.llm_client.OpenAI") as m:
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
#  0. Prompt sizing — pin per-subagent length so future bloat shows up
# ══════════════════════════════════════════════════════════════════

class TestPromptSizing:
    """Pin the per-subagent system-prompt sizes with ~25% headroom
    above the current length. Tighten when you intentionally shrink
    a prompt; loosen only if growth is genuinely needed."""

    def test_researcher_under_threshold(self):
        from agents.claude_style_v2.prompts import build_researcher_system_prompt
        prompt = build_researcher_system_prompt(_make_challenge())
        assert len(prompt) < 4_500, (
            f"researcher prompt grew to {len(prompt)} chars"
        )

    def test_designer_under_threshold(self):
        from agents.claude_style_v2.prompts import build_designer_system_prompt
        prompt = build_designer_system_prompt(_make_challenge())
        assert len(prompt) < 8_000, (
            f"designer prompt grew to {len(prompt)} chars"
        )

    def test_critic_under_threshold(self):
        from agents.claude_style_v2.prompts import build_critic_system_prompt
        prompt = build_critic_system_prompt()
        assert len(prompt) < 1_500, (
            f"critic prompt grew to {len(prompt)} chars"
        )


# ══════════════════════════════════════════════════════════════════
#  1. tools.build_tools — role parameter splits the surface
# ══════════════════════════════════════════════════════════════════

class TestRoleToolSubsets:
    def test_researcher_role_returns_research_tools(self):
        from agents.claude_style_v2.tools import build_tools
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
        from agents.claude_style_v2.tools import build_tools
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
        from agents.claude_style_v2.tools import build_tools
        with pytest.raises(ValueError):
            build_tools(_make_challenge(), role="potato")

    def test_no_role_returns_full_surface(self):
        from agents.claude_style_v2.tools import build_tools
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
        from agents.claude_style_v2.tools import TOOLS, build_tools
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
        from agents.claude_style_v2.hooks import (
            SUBMIT_BLOCKED_MSG, submit_requires_recent_validate,
        )
        assert (
            submit_requires_recent_validate("submit", {}, {})
            == SUBMIT_BLOCKED_MSG
        )

    def test_allows_when_recent_ok(self):
        from agents.claude_style_v2.hooks import submit_requires_recent_validate
        state = {"validate_history": [{"round": 5, "ok": True}]}
        assert (
            submit_requires_recent_validate("submit", {}, state) is None
        )

    def test_blocks_when_recent_failures_only(self):
        from agents.claude_style_v2.hooks import (
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
        from agents.claude_style_v2.hooks import (
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
        from agents.claude_style_v2.hooks import submit_requires_recent_validate
        # Hook is keyed on submit; everything else returns None.
        assert (
            submit_requires_recent_validate("validate_code", {}, {})
            is None
        )


class TestSubmitTimeGate:
    """v2 submit is time-gated: outside the last 5 minutes a submit
    stashes as best-so-far, inside it ships via SubmitSignal."""

    def test_late_window_ships(self):
        import time as _time
        from agents.claude_style_v2.tools import (
            SubmitSignal, build_handlers,
        )
        ch = _make_challenge()
        handlers = build_handlers(
            ch, deadline=_time.monotonic() + 60,
        )
        with pytest.raises(SubmitSignal) as excinfo:
            handlers["submit"](
                code=VALID_CODE, name="m", motivation="late",
            )
        assert excinfo.value.code == VALID_CODE

    def test_early_window_stashes(self):
        import time as _time
        from agents.claude_style_v2.tools import (
            EARLY_SUBMIT_GATE_SECONDS, build_handlers,
        )
        ch = _make_challenge()
        handlers = build_handlers(
            ch,
            deadline=_time.monotonic() + EARLY_SUBMIT_GATE_SECONDS + 600,
        )
        result = handlers["submit"](
            code=VALID_CODE, name="m", motivation="early try",
        )
        assert "stashed" in result.lower() or "best-so-far" in result.lower()
        sh = handlers["submit"]._state_holder
        best = sh["state"].get("best_so_far") or {}
        assert best.get("code") == VALID_CODE
        assert best.get("name") == "m"
        assert best.get("motivation") == "early try"


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
        from agents.claude_style_v2.subagents.researcher import run_researcher
        from agents.claude_style_v2.tools import build_handlers
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
        from agents.claude_style_v2.subagents.researcher import run_researcher
        from agents.claude_style_v2.tools import build_handlers
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
        from agents.claude_style_v2.subagents.researcher import run_researcher
        from agents.claude_style_v2.tools import build_handlers
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
        from agents.claude_style_v2.subagents.researcher import run_researcher
        from agents.claude_style_v2.tools import build_handlers
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
        from agents.claude_style_v2.subagents.researcher import run_researcher
        from agents.claude_style_v2.tools import build_handlers
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
        from agents.claude_style_v2 import agent
        # Researcher returns a brief; designer is still a stub
        # (returns None), so the orchestrator falls back.
        mock_openai.return_value.chat.completions.create.return_value = (
            _brief_completion(GOOD_BRIEF)
        )
        ch = _make_challenge()
        ch["agent_seconds"] = 180
        result = agent.design_architecture(ch, gated_client=None)
        assert "code" in result and "name" in result and "motivation" in result
        # Designer with no submit returned None, so we expect a fallback
        # template name (or auto_submit_*).
        assert (
            result["name"].startswith("fallback_")
            or result["name"].startswith("auto_submit_")
        )


# ══════════════════════════════════════════════════════════════════
#  5. Designer + critic + integration
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


def _designer_brief() -> dict:
    """Minimal brief for designer-only tests."""
    return {
        "relevant_prior_work": [],
        "frontier_gaps": [],
        "ideas_to_try": ["small linear baseline"],
        "plan": ["sketch a candidate", "validate", "submit"],
    }


class TestDesigner:
    def test_designer_validates_then_submits(self, mock_openai):
        """Happy path: validate_code → ok, submit ships, designer
        returns the SubmitSignal."""
        from agents.claude_style_v2.subagents.designer import run_designer
        from agents.claude_style_v2.tools import build_handlers

        # Sequence: round 1 = validate_code(VALID_CODE);
        # round 2 (after critic injection) = submit(candidate_id=...).
        # The mock returns the same critic completion regardless of
        # whether the call had `tools=` set, so we provide a generic
        # text completion for the critic in between.
        ch = _make_challenge(with_flops=False)

        call_log: list[dict] = []

        def chat_side_effect(*args, **kwargs):
            call_log.append(kwargs)
            n = len(call_log)
            if n == 1:
                return _make_completion(tool_calls=[
                    _make_tool_call(
                        "validate_code", {"code": VALID_CODE}, "v1",
                    ),
                ])
            if n == 2:
                # Critic call (no tools= kwarg).
                return _make_completion(content=(
                    "KEEP: clean baseline\nCHANGE: nothing\n"
                    "DROP: nothing"
                ))
            # Round 3 designer turn — submit the validated candidate.
            handlers_local = call_log[0].get("_handlers")
            return _make_completion(tool_calls=[
                _make_tool_call(
                    "submit",
                    {
                        "candidate_id": "PLACEHOLDER",
                        "name": "linear_baseline",
                        "motivation": "validated",
                        "note": "ok",
                    },
                    "s1",
                ),
            ])

        mock_openai.return_value.chat.completions.create.side_effect = (
            chat_side_effect
        )

        handlers = build_handlers(ch)
        # Pre-validate the code so the candidate exists in state and
        # the LLM's submit(candidate_id=...) resolves cleanly. We
        # also avoid the placeholder logic by calling validate_code
        # directly here and then patching the side_effect to use the
        # real id.
        cid = _validate_and_get_cid(handlers)

        # Re-wire side_effect to send the real cid.
        def chat_side_effect_v2(*args, **kwargs):
            call_log.append(kwargs)
            n = len(call_log)
            if n == 1:
                return _make_completion(tool_calls=[
                    _make_tool_call(
                        "validate_code",
                        {"candidate_id": cid},
                        "v1",
                    ),
                ])
            if n == 2:
                return _make_completion(content=(
                    "KEEP: clean baseline\nCHANGE: nothing\n"
                    "DROP: nothing"
                ))
            return _make_completion(tool_calls=[
                _make_tool_call(
                    "submit",
                    {
                        "candidate_id": cid,
                        "name": "linear_baseline",
                        "motivation": "validated",
                        "note": "ok",
                    },
                    "s1",
                ),
            ])

        call_log.clear()
        mock_openai.return_value.chat.completions.create.side_effect = (
            chat_side_effect_v2
        )

        import time as _time
        sig = run_designer(
            challenge=ch,
            handlers=handlers,
            deadline=_time.monotonic() + 120,
            llm_kwargs={
                "llm_url": "http://test/llm", "agent_token": "t",
                "miner_uid": "0", "model": "test-model",
                "temperature": 0.7, "max_tokens": 1024,
            },
            brief=_designer_brief(),
            state={},
            bucket="small",
        )
        assert sig is not None
        assert sig.name == "linear_baseline"
        assert sig.code == VALID_CODE
        # 1 validate + 1 critic + 1 submit = 3 chat calls.
        assert len(call_log) == 3

    def test_designer_blocked_submit_without_validate(self, mock_openai):
        """The submit_requires_recent_validate hook blocks submit when
        no recent validate_code returned ok=true. The designer should
        keep going (the LLM sees the blocked message) and never set
        submit_sig."""
        from agents.claude_style_v2.subagents.designer import run_designer
        from agents.claude_style_v2.tools import build_handlers

        # Sequence: round 1 = submit (BLOCKED) → loop continues, round
        # 2 = no tool call (assistant text only) → loop ends.
        completions = [
            _make_completion(tool_calls=[
                _make_tool_call(
                    "submit",
                    {"code": VALID_CODE, "name": "n", "motivation": "m"},
                    "s1",
                ),
            ]),
            _make_completion(content="giving up"),
        ]
        it = iter(completions)
        mock_openai.return_value.chat.completions.create.side_effect = (
            lambda *a, **kw: next(it)
        )

        ch = _make_challenge()
        import time as _time
        sig = run_designer(
            challenge=ch,
            handlers=build_handlers(ch),
            deadline=_time.monotonic() + 60,
            llm_kwargs={
                "llm_url": "http://test/llm", "agent_token": "t",
                "miner_uid": "0", "model": "test-model",
                "temperature": 0.7, "max_tokens": 1024,
            },
            brief=_designer_brief(),
            state={},
            bucket="small",
        )
        assert sig is None  # Hook blocked the submit.

    def test_designer_critic_runs_after_validate(self, mock_openai):
        """The critic must fire as a post-tool callback after each
        validate_code, and its critique must be injected as a
        user-role message into the next designer turn."""
        from agents.claude_style_v2.subagents.designer import run_designer
        from agents.claude_style_v2.tools import build_handlers

        ch = _make_challenge(with_flops=False)
        captured: list[dict] = []
        # We pre-validate to get a real candidate_id the LLM can ship.
        handlers = build_handlers(ch)
        cid = _validate_and_get_cid(handlers)
        critique_text = (
            "KEEP: linear backbone\nCHANGE: nothing\nDROP: nothing"
        )

        def chat_side_effect(*args, **kwargs):
            captured.append({
                "tools": kwargs.get("tools"),
                "messages": [dict(m) for m in kwargs.get("messages") or []],
            })
            n = len(captured)
            if n == 1:
                return _make_completion(tool_calls=[
                    _make_tool_call(
                        "validate_code",
                        {"candidate_id": cid},
                        "v1",
                    ),
                ])
            if n == 2:
                return _make_completion(content=critique_text)
            return _make_completion(tool_calls=[
                _make_tool_call(
                    "submit",
                    {
                        "candidate_id": cid,
                        "name": "n",
                        "motivation": "m",
                        "note": "ok",
                    },
                    "s1",
                ),
            ])

        mock_openai.return_value.chat.completions.create.side_effect = (
            chat_side_effect
        )

        import time as _time
        sig = run_designer(
            challenge=ch,
            handlers=handlers,
            deadline=_time.monotonic() + 60,
            llm_kwargs={
                "llm_url": "http://test/llm", "agent_token": "t",
                "miner_uid": "0", "model": "test-model",
                "temperature": 0.7, "max_tokens": 1024,
            },
            brief=_designer_brief(),
            state={},
            bucket="small",
        )
        assert sig is not None

        # Critic call: no tools= kwarg (single text completion).
        critic_call = captured[1]
        assert (critic_call["tools"] is None
                or critic_call["tools"] == [])

        # Round 3 (the post-critic designer turn) must see the
        # critique injected as a user-role message.
        third = captured[2]["messages"]
        critic_msgs = [
            m for m in third
            if m.get("role") == "user"
            and "Critic feedback" in str(m.get("content", ""))
        ]
        assert len(critic_msgs) == 1
        assert "KEEP" in critic_msgs[0]["content"]


def _validate_and_get_cid(handlers) -> str:
    """Helper: validate VALID_CODE, return the candidate_id stored."""
    out = handlers["validate_code"](code=VALID_CODE)
    cid_lines = [
        l for l in out.splitlines() if l.startswith("candidate_id: ")
    ]
    return cid_lines[-1].split(": ", 1)[1]


# ══════════════════════════════════════════════════════════════════
#  6. Orchestrator integration — routing / deadline / scratchpad
# ══════════════════════════════════════════════════════════════════

class TestOrchestratorIntegration:
    def test_routes_researcher_then_designer_then_submit(
        self, mock_openai,
    ):
        """Full happy path: researcher returns brief → designer
        validates and submits → orchestrator packages the submission."""
        from agents.claude_style_v2 import agent

        call_log: list[dict] = []

        def chat_side_effect(*args, **kwargs):
            call_log.append(kwargs)
            n = len(call_log)
            # Round 1: researcher returns the brief in a fenced block.
            if n == 1:
                return _make_completion(content=(
                    "```json\n" + json.dumps(GOOD_BRIEF) + "\n```"
                ))
            # Round 2: designer validates the code.
            if n == 2:
                return _make_completion(tool_calls=[
                    _make_tool_call(
                        "validate_code", {"code": VALID_CODE}, "v1",
                    ),
                ])
            # Round 3: critic call.
            if n == 3:
                return _make_completion(content=(
                    "KEEP: x\nCHANGE: nothing\nDROP: nothing"
                ))
            # Round 4: designer submits.
            return _make_completion(tool_calls=[
                _make_tool_call(
                    "submit",
                    {
                        "code": VALID_CODE,
                        "name": "shipped",
                        "motivation": "happy path",
                        "note": "n/a",
                    },
                    "s1",
                ),
            ])

        mock_openai.return_value.chat.completions.create.side_effect = (
            chat_side_effect
        )
        ch = _make_challenge(with_flops=False)
        # Keep agent_seconds short enough that the designer's deadline
        # falls inside the late-window (< EARLY_SUBMIT_GATE_SECONDS=300)
        # — but big enough that researcher gets > min_round_seconds=30s
        # of budget after the 0.15 fraction.
        ch["agent_seconds"] = 280
        result = agent.design_architecture(ch, gated_client=None)
        assert result["name"] == "shipped"
        assert result["code"] == VALID_CODE
        # 1 researcher + 1 validate + 1 critic + 1 submit = 4 calls.
        assert len(call_log) == 4

    def test_long_budget_early_submit_recovers_via_stash(self, mock_openai):
        """Long budget: designer submits early → submit stashes
        best_so_far → designer never re-submits → orchestrator's
        recovery branch ships the stashed candidate (NOT fallback)."""
        from agents.claude_style_v2 import agent

        call_log: list[dict] = []

        def chat_side_effect(*args, **kwargs):
            call_log.append(kwargs)
            n = len(call_log)
            # Round 1: researcher brief.
            if n == 1:
                return _make_completion(content=(
                    "```json\n" + json.dumps(GOOD_BRIEF) + "\n```"
                ))
            # Round 2: designer validates.
            if n == 2:
                return _make_completion(tool_calls=[
                    _make_tool_call(
                        "validate_code", {"code": VALID_CODE}, "v1",
                    ),
                ])
            # Round 3: critic.
            if n == 3:
                return _make_completion(content=(
                    "KEEP: x\nCHANGE: nothing\nDROP: nothing"
                ))
            # Round 4: designer submits — but budget is large so this
            # STASHES rather than shipping.
            if n == 4:
                return _make_completion(tool_calls=[
                    _make_tool_call(
                        "submit",
                        {
                            "code": VALID_CODE,
                            "name": "stashed_pick",
                            "motivation": "early try",
                            "note": "n/a",
                        },
                        "s1",
                    ),
                ])
            # Round 5+: designer gives up (no tool calls) → loop ends.
            return _make_completion(content="thinking, no further code")

        mock_openai.return_value.chat.completions.create.side_effect = (
            chat_side_effect
        )
        ch = _make_challenge(with_flops=False)
        # Big enough budget that designer_deadline is well outside the
        # late-window: budget * (researcher_frac + designer_frac) =
        # 1200 * 0.95 = 1140s out → far above EARLY_SUBMIT_GATE_SECONDS.
        ch["agent_seconds"] = 1200
        result = agent.design_architecture(ch, gated_client=None)
        # Recovery branch shipped the stashed candidate.
        assert result["name"] == "stashed_pick"
        assert result["code"] == VALID_CODE

    def test_short_budget_skips_to_fallback(self, mock_openai):
        """With less budget than the reserve window, the orchestrator
        should bypass the subagents entirely and return a fallback
        template — never calling chat."""
        from agents.claude_style_v2 import agent
        # Anything above zero, but well under FALLBACK_RESERVE_SECONDS
        # so the deadline is already in the past when we kick off.
        ch = _make_challenge()
        ch["agent_seconds"] = 1
        result = agent.design_architecture(ch, gated_client=None)
        # Fallback template was returned.
        assert result["name"].startswith("fallback_")
        assert "build_model" in result["code"]
        # No chat call ever fired.
        assert (
            mock_openai.return_value.chat.completions.create.call_count
            == 0
        )

    def test_designer_failure_falls_through_to_fallback(self, mock_openai):
        """Researcher returns brief → designer can't ship (LLM only
        returns prose) → orchestrator falls back to the template."""
        from agents.claude_style_v2 import agent

        completions = [
            # Round 1: researcher brief.
            _make_completion(content=(
                "```json\n" + json.dumps(GOOD_BRIEF) + "\n```"
            )),
            # Designer rounds: prose, no tool calls — loop exits
            # immediately, no submit.
            _make_completion(content="cannot decide"),
        ]
        it = iter(completions)
        mock_openai.return_value.chat.completions.create.side_effect = (
            lambda *a, **kw: next(it)
        )

        ch = _make_challenge()
        ch["agent_seconds"] = 240
        result = agent.design_architecture(ch, gated_client=None)
        assert result["name"].startswith("fallback_")
        assert "build_model" in result["code"]
        assert "FALLBACK" in result["motivation"]

    def test_scratchpad_state_survives_across_calls(
        self, mock_openai, tmp_path,
    ):
        """A submit on round N writes state to scratch_dir; round N+1
        loads the same scratch_dir and sees the prior submission in
        history. We simulate the harness by injecting
        load_scratchpad/save_scratchpad into the agent module."""
        from agents.claude_style_v2 import agent
        # Patch in the harness-style scratchpad globals.
        monkey_dir = str(tmp_path)

        def _load(_challenge):
            return monkey_dir

        def _save(_challenge, _scratch_dir):
            return None

        original_globals = dict(agent.__dict__)
        agent.__dict__["load_scratchpad"] = _load
        agent.__dict__["save_scratchpad"] = _save
        try:
            # Round 1: submit succeeds.
            def chat_round1(*args, **kwargs):
                n = chat_round1.calls = chat_round1.calls + 1 if hasattr(
                    chat_round1, "calls",
                ) else 1
                if n == 1:
                    return _make_completion(content=(
                        "```json\n" + json.dumps(GOOD_BRIEF) + "\n```"
                    ))
                if n == 2:
                    return _make_completion(tool_calls=[
                        _make_tool_call(
                            "validate_code", {"code": VALID_CODE},
                            "v1",
                        ),
                    ])
                if n == 3:
                    return _make_completion(content=(
                        "KEEP: x\nCHANGE: y\nDROP: z"
                    ))
                return _make_completion(tool_calls=[
                    _make_tool_call(
                        "submit",
                        {
                            "code": VALID_CODE,
                            "name": "round1",
                            "motivation": "first",
                        },
                        "s1",
                    ),
                ])

            mock_openai.return_value.chat.completions.create.side_effect = (
                chat_round1
            )
            ch1 = _make_challenge(with_flops=False)
            # Keep budget inside the late-window so the first submit
            # actually ships (only one submission recorded). Larger
            # budgets would stash and re-submit on every loop turn,
            # appending to the submissions list each time.
            ch1["agent_seconds"] = 280
            ch1["round_id"] = 1
            r1 = agent.design_architecture(ch1, gated_client=None)
            assert r1["name"] == "round1"

            # Round 2: a fresh design_architecture call sees the prior
            # submission persisted on disk.
            from agents.claude_style_v2.core.history import (
                load_state, get_submissions,
            )
            state2 = load_state(monkey_dir)
            subs = get_submissions(state2)
            assert len(subs) == 1
            assert subs[0]["name"] == "round1"
        finally:
            # Undo our globals injection.
            for k in ("load_scratchpad", "save_scratchpad"):
                if k not in original_globals:
                    agent.__dict__.pop(k, None)


# ══════════════════════════════════════════════════════════════════
#  7. Smoke test — saved challenge.json round-trip
# ══════════════════════════════════════════════════════════════════

class TestSmoke:
    """End-to-end smoke test that mirrors what the harness does:
    load a challenge.json, run design_architecture with a mock
    client, confirm the returned dict carries valid Python with
    top-level build_model and build_optimizer defs.

    The challenge.json is hand-rolled inside the test (no on-disk
    fixture file) so the test is self-contained.
    """

    def test_returns_ast_parseable_with_required_defs(
        self, mock_openai, tmp_path,
    ):
        import ast
        from agents.claude_style_v2 import agent

        challenge = {
            "challenge_id": "smoke-1",
            "round_id": 99,
            "agent_seconds": 240,
            "min_flops_equivalent": 500_000,
            "max_flops_equivalent": 2_000_000,
            "task": {
                "name": "ts_forecasting",
                "description": "Time series forecasting smoke test",
                "task_params": {
                    "context_len": 256,
                    "prediction_len": 48,
                    "num_variates": 1,
                    "quantiles": [0.1, 0.5, 0.9],
                },
                "constraints": [
                    "Output shape must be (batch, prediction_len, "
                    "num_variates, len(quantiles))"
                ],
                "objectives": [{"name": "crps", "primary": True}],
            },
            "feasible_frontier": [],
        }
        # Persist it to disk and reload — matches "load a saved
        # challenge.json" in the spec.
        path = tmp_path / "challenge.json"
        path.write_text(json.dumps(challenge))
        loaded = json.loads(path.read_text())

        # Mock the LLM to return prose only — designer can't ship,
        # orchestrator must fall through to the fallback template.
        # The fallback path is the strongest smoke test because it
        # exercises orchestrator + scratchpad + fallback packaging
        # without depending on LLM cooperation.
        mock_openai.return_value.chat.completions.create.return_value = (
            _make_completion(content="no JSON here")
        )

        result = agent.design_architecture(loaded, gated_client=None)
        assert set(result.keys()) == {"code", "name", "motivation"}
        assert isinstance(result["code"], str)
        assert isinstance(result["name"], str)
        assert isinstance(result["motivation"], str)
        # Code is AST-parseable.
        tree = ast.parse(result["code"])
        # Top-level build_model + build_optimizer defs present.
        top_level = {
            n.name for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert "build_model" in top_level
        assert "build_optimizer" in top_level
