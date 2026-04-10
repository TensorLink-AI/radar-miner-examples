"""Tests for reliable_baseline agent — template validity, never-empty guarantee,
validation correctness, LLM error handling, and prompt building."""

import importlib.util
import json
import os
import sys
import tempfile

# ── Module loading ────────────────────────────────────────────────
_agent_dir = os.path.join(os.path.dirname(__file__), "..", "agents", "reliable_baseline")
sys.path.insert(0, _agent_dir)

# Clear cached core modules so we pick up reliable_baseline's core
for _k in list(sys.modules.keys()):
    if _k == "core" or _k.startswith("core."):
        del sys.modules[_k]

_spec = importlib.util.spec_from_file_location(
    "reliable_baseline_agent", os.path.join(_agent_dir, "agent.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

design_architecture = _mod.design_architecture

from core.validation import validate_code
from core.llm import extract_code, reason_and_generate, MAX_LLM_ATTEMPTS
from core.prompt_builder import (
    build_system_prompt, build_user_prompt,
)
from core.history import (
    identify_bucket, extract_flops_budget, load_state, save_state,
    add_entry, get_history, format_history,
)
from core.db_client import recent_experiments, recent_failures, component_stats, dead_ends
from core.templates import generate_fallback_code


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
        return self.responses.get(url, {})

    def post_json(self, url, payload):
        self.calls.append(("POST", url, payload))
        if self.raise_on_call:
            raise RuntimeError("mock error")
        return self.responses.get(url, {})


def _make_challenge(bucket="small", llm_url="", db_url="", frontier=None):
    """Create a minimal challenge dict for a given bucket."""
    from core.history import SIZE_BUCKETS
    flops_min, flops_max = SIZE_BUCKETS.get(bucket, (500_000, 2_000_000))
    return {
        "min_flops_equivalent": flops_min,
        "max_flops_equivalent": flops_max,
        "feasible_frontier": frontier or [],
        "task": {"task_params": {
            "context_len": 512, "prediction_len": 96,
            "num_variates": 1,
            "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }},
        "db_url": db_url,
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


# ══════════════════════════════════════════════════════════════════
#  1. Dynamic fallback code generation
# ══════════════════════════════════════════════════════════════════

class TestFallbackCodeGeneration:
    def test_templates_module_has_no_old_templates(self):
        """templates.py should not export old hardcoded templates."""
        import importlib
        tmpl_path = os.path.join(_agent_dir, "core", "templates.py")
        spec = importlib.util.spec_from_file_location("core.templates_check", tmpl_path)
        tmpl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tmpl)
        assert not hasattr(tmpl, "get_template"), "get_template should be removed"
        assert not hasattr(tmpl, "_TEMPLATES"), "_TEMPLATES should be removed"

    def test_fallback_generates_valid_code(self):
        """Fallback generates code that passes validation."""
        challenge = _make_challenge("small")
        code = generate_fallback_code(challenge)
        assert code, "Fallback should produce non-empty code"
        ok, errors = validate_code(code, challenge)
        assert ok, f"Fallback code failed validation: {errors}"

    def test_fallback_has_build_model(self):
        """Fallback code contains build_model with correct params."""
        challenge = _make_challenge("medium")
        code = generate_fallback_code(challenge)
        assert "def build_model(" in code
        assert "context_len" in code
        assert "prediction_len" in code

    def test_fallback_has_build_optimizer(self):
        """Fallback code contains build_optimizer."""
        challenge = _make_challenge("small")
        code = generate_fallback_code(challenge)
        assert "def build_optimizer(model)" in code

    def test_fallback_every_bucket(self):
        """Fallback produces valid code for every bucket."""
        from core.history import SIZE_BUCKETS
        for bucket in SIZE_BUCKETS:
            challenge = _make_challenge(bucket)
            code = generate_fallback_code(challenge)
            assert code, f"No fallback for {bucket}"
            ok, errors = validate_code(code, challenge)
            assert ok, f"Fallback invalid for {bucket}: {errors}"

    def test_fallback_empty_without_task_params(self):
        """No task_params means no fallback possible."""
        challenge = {"task": {}, "min_flops_equivalent": 0, "max_flops_equivalent": 0}
        code = generate_fallback_code(challenge)
        assert code == ""


# ══════════════════════════════════════════════════════════════════
#  2. Graceful degradation — design_architecture does not crash without LLM
# ══════════════════════════════════════════════════════════════════

class TestGracefulDegradation:
    """Without LLM, design_architecture uses dynamic fallback code."""

    def _run(self, challenge, client):
        """Run design_architecture with injected globals."""
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

    def test_no_llm_returns_fallback_code(self):
        """No LLM URL — returns fallback code (not empty)."""
        challenge = _make_challenge("small", llm_url="", db_url="")
        client = MockClient(raise_on_call=True)
        result = self._run(challenge, client)
        assert result["code"] != "", "Should return fallback code, not empty"
        assert "build_model" in result["code"]
        assert "fallback" in result["name"]

    def test_llm_failure_returns_fallback_code(self):
        """LLM raises — returns fallback code (not empty)."""
        challenge = _make_challenge("small", llm_url="http://fake-llm",
                                    db_url="http://fake-db")
        client = MockClient(raise_on_call=True)
        result = self._run(challenge, client)
        assert result["code"] != "", "Should return fallback code, not empty"
        assert "build_model" in result["code"]

    def test_does_not_crash_without_llm(self):
        """Agent returns a dict with valid code when LLM unavailable."""
        challenge = _make_challenge("medium", llm_url="", frontier=[])
        client = MockClient()
        result = self._run(challenge, client)
        assert isinstance(result, dict)
        assert "code" in result
        assert "name" in result
        assert result["code"] != ""

    def test_does_not_crash_with_frontier(self):
        """Agent handles frontier context without LLM gracefully."""
        frontier = [{"objectives": {"crps": 0.4}, "code": "x=1"}]
        challenge = _make_challenge("large", llm_url="", frontier=frontier)
        client = MockClient()
        result = self._run(challenge, client)
        assert isinstance(result, dict)
        assert result["code"] != ""

    def test_every_bucket_produces_valid_code(self):
        """Every bucket produces valid fallback code when LLM is unavailable."""
        from core.history import SIZE_BUCKETS
        for bucket in SIZE_BUCKETS:
            challenge = _make_challenge(bucket, llm_url="")
            client = MockClient()
            result = self._run(challenge, client)
            assert isinstance(result, dict), f"Non-dict result for bucket {bucket}"
            assert result["code"] != "", f"Empty code for bucket {bucket}"
            ok, errors = validate_code(result["code"])
            assert ok, f"Invalid fallback for {bucket}: {errors}"


# ══════════════════════════════════════════════════════════════════
#  4. Validation correctness
# ══════════════════════════════════════════════════════════════════

class TestValidation:
    def test_valid_code_passes(self):
        ok, errors = validate_code(VALID_CODE)
        assert ok, f"Valid code rejected: {errors}"

    def test_empty_code_rejected(self):
        ok, errors = validate_code("")
        assert not ok
        assert any("Empty" in e or "empty" in e.lower() for e in errors)

    def test_whitespace_only_rejected(self):
        ok, errors = validate_code("   \n\n  ")
        assert not ok

    def test_syntax_error_rejected(self):
        ok, errors = validate_code("def foo(:\n  pass")
        assert not ok
        assert any("SyntaxError" in e for e in errors)

    def test_missing_build_model_rejected(self):
        code = "def build_optimizer(model): pass"
        ok, errors = validate_code(code)
        assert not ok
        assert any("build_model" in e for e in errors)

    def test_missing_build_optimizer_rejected(self):
        code = "def build_model(context_len, prediction_len, num_variates, quantiles): pass"
        ok, errors = validate_code(code)
        assert not ok
        assert any("build_optimizer" in e for e in errors)

    def test_missing_params_rejected(self):
        challenge = _make_challenge("small")
        code = "def build_model(x): pass\ndef build_optimizer(model): pass"
        ok, errors = validate_code(code, challenge)
        assert not ok
        assert any("missing parameter" in e for e in errors)

    def test_forbidden_import_subprocess(self):
        code = (
            "import subprocess\n"
            "def build_model(context_len, prediction_len, num_variates, quantiles): pass\n"
            "def build_optimizer(model): pass"
        )
        ok, errors = validate_code(code)
        assert not ok
        assert any("subprocess" in e for e in errors)

    def test_forbidden_import_socket(self):
        code = (
            "import socket\n"
            "def build_model(context_len, prediction_len, num_variates, quantiles): pass\n"
            "def build_optimizer(model): pass"
        )
        ok, errors = validate_code(code)
        assert not ok
        assert any("socket" in e for e in errors)

    def test_forbidden_import_from(self):
        code = (
            "from subprocess import run\n"
            "def build_model(context_len, prediction_len, num_variates, quantiles): pass\n"
            "def build_optimizer(model): pass"
        )
        ok, errors = validate_code(code)
        assert not ok
        assert any("subprocess" in e for e in errors)

    def test_build_model_inside_class_rejected(self):
        """Functions inside a class are NOT top-level and should be rejected."""
        code = (
            "class Foo:\n"
            "    def build_model(self, context_len, prediction_len, num_variates, quantiles): pass\n"
            "    def build_optimizer(self, model): pass\n"
        )
        ok, errors = validate_code(code)
        assert not ok, "Functions inside a class should not count as top-level"


# ══════════════════════════════════════════════════════════════════
#  5. LLM error handling
# ══════════════════════════════════════════════════════════════════

class TestLLMErrorHandling:
    def test_extract_code_python_block(self):
        text = "Here is code:\n```python\nx = 1\n```\nDone."
        assert extract_code(text) == "x = 1"

    def test_extract_code_bare_block(self):
        text = "```\nx = 2\n```"
        assert extract_code(text) == "x = 2"

    def test_extract_code_no_block(self):
        text = "No code here, just text."
        assert extract_code(text) == ""

    def test_extract_code_truncated(self):
        text = "```python\nx = 3\nmore code"
        assert "x = 3" in extract_code(text)

    def test_extract_code_raw_python(self):
        """Raw Python with build_model should be extracted."""
        text = "import torch\ndef build_model(): pass"
        assert extract_code(text) != ""

    def test_reason_and_generate_returns_none_on_garbage(self):
        """LLM returns garbage text — reason_and_generate returns None."""
        client = MockClient(responses={
            "http://llm/v1/chat/completions": {
                "choices": [{"message": {"content": "I don't know how to code"}}]
            }
        })
        challenge = _make_challenge("small", llm_url="http://llm")
        context = {"frontier": [], "recent_experiments": {}, "failures": {},
                   "component_stats": {}, "dead_ends": {}, "history": [],
                   "bucket": "small", "target_flops": 1_100_000}
        result = reason_and_generate(client, challenge, context)
        assert result is None

    def test_reason_and_generate_returns_none_on_empty(self):
        """LLM returns empty response — reason_and_generate returns None."""
        client = MockClient(responses={
            "http://llm/v1/chat/completions": {
                "choices": [{"message": {"content": ""}}]
            }
        })
        challenge = _make_challenge("small", llm_url="http://llm")
        context = {"frontier": [], "recent_experiments": {}, "failures": {},
                   "component_stats": {}, "dead_ends": {}, "history": [],
                   "bucket": "small", "target_flops": 1_100_000}
        result = reason_and_generate(client, challenge, context)
        assert result is None

    def test_reason_and_generate_returns_none_on_exception(self):
        """LLM raises exception — reason_and_generate returns None."""
        client = MockClient(raise_on_call=True)
        challenge = _make_challenge("small", llm_url="http://llm")
        context = {"frontier": [], "recent_experiments": {}, "failures": {},
                   "component_stats": {}, "dead_ends": {}, "history": [],
                   "bucket": "small", "target_flops": 1_100_000}
        result = reason_and_generate(client, challenge, context)
        assert result is None

    def test_reason_and_generate_succeeds_with_valid_code(self):
        """LLM returns valid code — reason_and_generate returns it."""
        client = MockClient(responses={
            "http://llm/v1/chat/completions": {
                "choices": [{"message": {"content": f"```python\n{VALID_CODE}\n```"}}]
            }
        })
        # Use challenge without FLOPs budget so VALID_CODE passes structural
        # validation (this test is about LLM error handling, not FLOPs sizing)
        challenge = {
            "task": {"run_command": "python harness.py"},
            "feasible_frontier": [],
            "db_url": "",
            "llm_url": "http://llm",
            "seed": 42,
            "round_id": 1,
        }
        context = {"frontier": [], "recent_experiments": {}, "failures": {},
                   "component_stats": {}, "dead_ends": {}, "history": [],
                   "bucket": "small", "target_flops": 1_100_000}
        result = reason_and_generate(client, challenge, context)
        assert result is not None
        code, name, motivation = result
        ok, errors = validate_code(code)
        assert ok, f"LLM code failed validation: {errors}"

    def test_max_llm_attempts_is_fifteen(self):
        """Max LLM attempts should be 15 (half the 30-request rate limit)."""
        assert MAX_LLM_ATTEMPTS == 15

    def test_reason_and_generate_no_llm_url(self):
        """No LLM URL — returns None immediately."""
        client = MockClient()
        challenge = _make_challenge("small", llm_url="")
        context = {"frontier": [], "recent_experiments": {}, "failures": {},
                   "component_stats": {}, "dead_ends": {}, "history": [],
                   "bucket": "small", "target_flops": 1_100_000}
        result = reason_and_generate(client, challenge, context)
        assert result is None


# ══════════════════════════════════════════════════════════════════
#  6. Prompt building — correct values
# ══════════════════════════════════════════════════════════════════

class TestPromptBuilding:
    def test_system_prompt_contains_task_params(self):
        """System prompt includes task_params from the challenge."""
        challenge = _make_challenge("small")
        prompt = build_system_prompt(challenge)
        assert "context_len" in prompt
        assert "prediction_len" in prompt
        assert "build_model" in prompt
        assert "build_optimizer" in prompt

    def test_user_prompt_contains_dynamic_values(self):
        challenge = _make_challenge("small")
        challenge["task"] = {"task_params": {"num_variates": 1, "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}}
        context = {"frontier": [], "recent_experiments": {}, "failures": {},
                   "component_stats": {}, "dead_ends": {}, "history": [],
                   "bucket": "small", "target_flops": 1_100_000}
        prompt = build_user_prompt(challenge, context)
        assert "num_variates=1" in prompt
        assert "0.1, 0.2, 0.3" in prompt

    def test_user_prompt_contains_flops_budget(self):
        challenge = _make_challenge("medium")
        context = {"frontier": [], "recent_experiments": {}, "failures": {},
                   "component_stats": {}, "dead_ends": {}, "history": [],
                   "bucket": "medium", "target_flops": 27_500_000}
        prompt = build_user_prompt(challenge, context)
        assert "FLOPs" in prompt
        assert "medium" in prompt.lower()

    def test_system_prompt_includes_rules(self):
        challenge = _make_challenge("small")
        prompt = build_system_prompt(challenge)
        assert "build_model" in prompt
        assert "build_optimizer" in prompt
        assert "subprocess" in prompt  # mentioned in forbidden list

    def test_user_prompt_with_frontier(self):
        frontier = [{"objectives": {"crps": 0.42, "mase": 0.55}, "code": "x=1"}]
        challenge = _make_challenge("small", frontier=frontier)
        context = {"frontier": frontier, "recent_experiments": {}, "failures": {},
                   "component_stats": {}, "dead_ends": {}, "history": [],
                   "bucket": "small", "target_flops": 1_100_000}
        prompt = build_user_prompt(challenge, context)
        assert "0.42" in prompt
        assert "frontier" in prompt.lower() or "Frontier" in prompt

    def test_user_prompt_contains_sizing_guidance(self):
        """User prompt should include FLOPs sizing guidance."""
        challenge = _make_challenge("small")
        context = {"frontier": [], "recent_experiments": {}, "failures": {},
                   "component_stats": {}, "dead_ends": {}, "history": [],
                   "bucket": "small", "target_flops": 1_100_000}
        prompt = build_user_prompt(challenge, context)
        assert "hidden" in prompt.lower()


# ══════════════════════════════════════════════════════════════════
#  7. History / scratchpad
# ══════════════════════════════════════════════════════════════════

class TestHistory:
    def test_identify_bucket(self):
        assert identify_bucket(100_000, 500_000) == "tiny"
        assert identify_bucket(500_000, 2_000_000) == "small"
        assert identify_bucket(2_000_000, 10_000_000) == "medium_small"
        assert identify_bucket(10_000_000, 50_000_000) == "medium"
        assert identify_bucket(50_000_000, 125_000_000) == "large"

    def test_identify_bucket_fuzzy(self):
        # Non-exact match should return closest
        bucket = identify_bucket(400_000, 600_000)
        assert bucket in ("tiny", "small")

    def test_extract_flops_flat(self):
        challenge = {"min_flops_equivalent": 500_000, "max_flops_equivalent": 2_000_000}
        fmin, fmax = extract_flops_budget(challenge)
        assert fmin == 500_000
        assert fmax == 2_000_000

    def test_extract_flops_nested(self):
        challenge = {"flops_budget": {"min": 100_000, "max": 500_000}}
        fmin, fmax = extract_flops_budget(challenge)
        assert fmin == 100_000
        assert fmax == 500_000

    def test_add_entry_and_get(self):
        state = {}
        state = add_entry(state, name="test", code="x=1",
                          motivation="testing", bucket="small")
        entries = get_history(state)
        assert len(entries) == 1
        assert entries[0]["name"] == "test"

    def test_history_limit(self):
        state = {}
        for i in range(60):
            state = add_entry(state, name=f"exp{i}", code=f"x={i}",
                              motivation="test")
        assert len(get_history(state)) == 50

    def test_format_history_empty(self):
        assert format_history([]) == "No previous submissions."

    def test_save_load_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {"history": [{"name": "test"}]}
            save_state(tmp, state)
            loaded = load_state(tmp)
            assert loaded == state

    def test_load_state_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert load_state(tmp) == {}


# ══════════════════════════════════════════════════════════════════
#  8. DB client error handling
# ══════════════════════════════════════════════════════════════════

class TestDBClient:
    def test_recent_experiments_error(self):
        client = MockClient(raise_on_call=True)
        result = recent_experiments(client, "http://fake")
        assert result == {}

    def test_recent_failures_error(self):
        client = MockClient(raise_on_call=True)
        result = recent_failures(client, "http://fake")
        assert result == {}

    def test_component_stats_error(self):
        client = MockClient(raise_on_call=True)
        result = component_stats(client, "http://fake")
        assert result == {}

    def test_dead_ends_error(self):
        client = MockClient(raise_on_call=True)
        result = dead_ends(client, "http://fake")
        assert result == {}

    def test_recent_experiments_success(self):
        client = MockClient(responses={
            "http://db/experiments/recent?n=15": {"experiments": [{"name": "exp1"}]}
        })
        result = recent_experiments(client, "http://db")
        assert "experiments" in result


# ══════════════════════════════════════════════════════════════════
#  9. Integration: extraction + validation pipeline
# ══════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_extract_and_validate_pipeline(self):
        """Code extraction from LLM output + validation works end-to-end."""
        raw = f"Here's the model:\n```python\n{VALID_CODE}\n```\nDone!"
        code = extract_code(raw)
        ok, errors = validate_code(code)
        assert ok, f"Pipeline failed: {errors}"

    def test_valid_code_extract_validate(self):
        """Valid code passes through the full validation pipeline."""
        ok, errors = validate_code(VALID_CODE)
        assert ok, f"Valid code pipeline failed: {errors}"
