"""Tests for the GEPA / random_mutate prompt-variant integration.

Every example agent must:

  1. Expose ``_load_active_prompt(round_id)`` that reads
     ``${MINER_PROMPTS_DIR}/active.json`` and returns
     ``{"id": str, "template": str}``. Missing / empty / malformed
     files return both fields empty so the agent gracefully falls back
     to its hardcoded behavior.
  2. Round-robin the population by ``round_id``.
  3. Round-trip ``prompt_id`` on the dict returned by
     ``design_architecture`` so Phase C scores attribute back to the
     prompt variant via ``experiments.prompt_id``.

These tests are loader / contract tests — they don't exercise the
LLM-driven design loop (the per-agent test files cover that).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest


AGENTS = [
    ("autonomous", "agents/autonomous"),
    ("openai_sdk", "agents/openai_sdk"),
    ("openai_sdk_v2", "agents/openai_sdk_v2"),
    ("claude_style", "agents/claude_style"),
    ("claude_style_v2", "agents/claude_style_v2"),
    ("patch_decoder", "agents/patch_decoder"),
]


_AGENT_LOCAL_MODULE_NAMES = {
    "core", "subagents", "prompts", "tools", "llm_client", "hooks",
    "validation", "strategies",
}


def _purge_agent_module_cache():
    """Drop any module that resolves to an agent's local sibling
    package so the next ``_load_agent`` resolves imports against
    THAT agent's directory, not whichever agent loaded last."""
    for k in list(sys.modules.keys()):
        head = k.split(".", 1)[0]
        if head in _AGENT_LOCAL_MODULE_NAMES:
            del sys.modules[k]


def _snapshot_agent_local_modules() -> dict:
    """Capture the current ``core`` / ``tools`` / ... bindings so we
    can restore them after the test. Other test modules cache these
    at collection time and assume they stay pinned to the right
    agent — this fixture's mid-run reloads would otherwise leave
    them dangling."""
    snap = {}
    for k, mod in sys.modules.items():
        head = k.split(".", 1)[0]
        if head in _AGENT_LOCAL_MODULE_NAMES:
            snap[k] = mod
    return snap


def _restore_agent_local_modules(snap: dict) -> None:
    _purge_agent_module_cache()
    sys.modules.update(snap)


def _load_agent(agent_dir: str) -> tuple:
    """Load an agent's ``agent.py`` as an isolated module so each
    agent's local ``core`` package wins on sys.path. Returns the
    module plus the absolute path we inserted, so the caller can
    remove just that entry on teardown without disturbing path
    additions other test modules made at collection time."""
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    full_dir = os.path.abspath(os.path.join(repo_root, agent_dir))
    _purge_agent_module_cache()
    sys.path.insert(0, full_dir)
    spec = importlib.util.spec_from_file_location(
        f"agent_under_test_{os.path.basename(full_dir)}",
        os.path.join(full_dir, "agent.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, full_dir


@pytest.fixture(params=AGENTS, ids=[name for name, _ in AGENTS])
def agent_mod(request):
    name, path = request.param
    snap = _snapshot_agent_local_modules()
    mod, inserted = _load_agent(path)
    yield mod
    try:
        sys.path.remove(inserted)
    except ValueError:
        pass
    _restore_agent_local_modules(snap)


@pytest.fixture(autouse=True)
def _isolate_module_cache():
    """Belt-and-braces: even tests in this file that don't use the
    parametrized fixture must leave the agent-local module bindings
    exactly as they found them — neighbour test files cache them at
    collection time and rely on those references staying live."""
    snap = _snapshot_agent_local_modules()
    yield
    _restore_agent_local_modules(snap)


def test_load_active_prompt_missing_dir_returns_empty(
    agent_mod, tmp_path, monkeypatch,
):
    monkeypatch.setenv("MINER_PROMPTS_DIR", str(tmp_path / "does_not_exist"))
    out = agent_mod._load_active_prompt(0)
    assert out == {"id": "", "template": ""}


def test_load_active_prompt_malformed_json_returns_empty(
    agent_mod, tmp_path, monkeypatch,
):
    (tmp_path / "active.json").write_text("not json {{{")
    monkeypatch.setenv("MINER_PROMPTS_DIR", str(tmp_path))
    out = agent_mod._load_active_prompt(0)
    assert out == {"id": "", "template": ""}


def test_load_active_prompt_empty_population_returns_empty(
    agent_mod, tmp_path, monkeypatch,
):
    (tmp_path / "active.json").write_text(json.dumps({"prompts": []}))
    monkeypatch.setenv("MINER_PROMPTS_DIR", str(tmp_path))
    out = agent_mod._load_active_prompt(0)
    assert out == {"id": "", "template": ""}


def test_load_active_prompt_picks_first_when_single(
    agent_mod, tmp_path, monkeypatch,
):
    (tmp_path / "active.json").write_text(json.dumps({
        "prompts": [
            {"id": "abc12345", "template": "Be terse."},
        ],
    }))
    monkeypatch.setenv("MINER_PROMPTS_DIR", str(tmp_path))
    for r in (0, 1, 7, 999):
        out = agent_mod._load_active_prompt(r)
        assert out == {"id": "abc12345", "template": "Be terse."}


def test_load_active_prompt_round_robins_population(
    agent_mod, tmp_path, monkeypatch,
):
    (tmp_path / "active.json").write_text(json.dumps({
        "prompts": [
            {"id": "id_a", "template": "A"},
            {"id": "id_b", "template": "B"},
            {"id": "id_c", "template": "C"},
        ],
    }))
    monkeypatch.setenv("MINER_PROMPTS_DIR", str(tmp_path))

    assert agent_mod._load_active_prompt(0)["id"] == "id_a"
    assert agent_mod._load_active_prompt(1)["id"] == "id_b"
    assert agent_mod._load_active_prompt(2)["id"] == "id_c"
    assert agent_mod._load_active_prompt(3)["id"] == "id_a"  # wraps
    assert agent_mod._load_active_prompt(7)["id"] == "id_b"


def test_load_active_prompt_accepts_bare_list(
    agent_mod, tmp_path, monkeypatch,
):
    """The optimizer writes ``{"prompts": [...]}`` but a hand-edited
    bare list should also work — keeps the format forgiving."""
    (tmp_path / "active.json").write_text(json.dumps([
        {"id": "id_x", "template": "X"},
    ]))
    monkeypatch.setenv("MINER_PROMPTS_DIR", str(tmp_path))
    assert agent_mod._load_active_prompt(0)["id"] == "id_x"


def test_load_active_prompt_coerces_non_string_fields(
    agent_mod, tmp_path, monkeypatch,
):
    (tmp_path / "active.json").write_text(json.dumps({
        "prompts": [{"id": 42, "template": None}],
    }))
    monkeypatch.setenv("MINER_PROMPTS_DIR", str(tmp_path))
    out = agent_mod._load_active_prompt(0)
    assert out["id"] == "42"
    assert out["template"] == "None"


# ── patch_decoder end-to-end (deterministic, no LLM needed) ────────

def test_patch_decoder_returns_prompt_id_when_population_present(
    tmp_path, monkeypatch,
):
    mod, _ = _load_agent("agents/patch_decoder")
    (tmp_path / "active.json").write_text(json.dumps({
        "prompts": [
            {"id": "pid_one", "template": "ignored by deterministic agent"},
        ],
    }))
    monkeypatch.setenv("MINER_PROMPTS_DIR", str(tmp_path))

    challenge = {
        "round_id": 5,
        "task": {
            "name": "ts_forecasting",
            "task_params": {
                "context_len": 96,
                "prediction_len": 24,
                "num_variates": 1,
                "quantiles": [0.1, 0.5, 0.9],
            },
            "time_budget": 60,
        },
        "min_flops_equivalent": 100_000,
        "max_flops_equivalent": 500_000,
        "feasible_frontier": [],
    }
    result = mod.design_architecture(challenge, client=None)
    assert "prompt_id" in result
    assert result["prompt_id"] == "pid_one"


def test_patch_decoder_returns_empty_prompt_id_without_population(
    tmp_path, monkeypatch,
):
    """When the miner hasn't run the optimizer, prompt_id is empty —
    Phase C falls back to whole-miner attribution, which is correct."""
    mod, _ = _load_agent("agents/patch_decoder")
    monkeypatch.setenv("MINER_PROMPTS_DIR", str(tmp_path / "no_prompts"))

    challenge = {
        "round_id": 0,
        "task": {
            "name": "ts_forecasting",
            "task_params": {
                "context_len": 96,
                "prediction_len": 24,
                "num_variates": 1,
                "quantiles": [0.1, 0.5, 0.9],
            },
            "time_budget": 60,
        },
        "min_flops_equivalent": 100_000,
        "max_flops_equivalent": 500_000,
        "feasible_frontier": [],
    }
    result = mod.design_architecture(challenge, client=None)
    assert "prompt_id" in result
    assert result["prompt_id"] == ""
