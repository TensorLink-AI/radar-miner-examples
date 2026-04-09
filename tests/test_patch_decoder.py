"""Tests for patch_decoder agent — deterministic code generation, validation,
output shapes, scaling correctness, and never-empty guarantee."""

import importlib.util
import os
import sys
import tempfile

# ── Module loading ────────────────────────────────────────────────
_agent_dir = os.path.join(os.path.dirname(__file__), "..", "agents", "patch_decoder")
sys.path.insert(0, _agent_dir)

# Clear cached core modules so we pick up patch_decoder's core
for _k in list(sys.modules.keys()):
    if _k == "core" or _k.startswith("core."):
        del sys.modules[_k]

_spec = importlib.util.spec_from_file_location(
    "patch_decoder_agent", os.path.join(_agent_dir, "agent.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

design_architecture = _mod.design_architecture
_generate_code = _mod._generate_code
_compute_scaling = _mod._compute_scaling

from core.validation import validate_code
from core.history import (
    identify_bucket, extract_flops_budget, SIZE_BUCKETS,
    load_state, save_state, add_entry, get_history,
)


def _scaling_for_bucket(bucket, num_variates=1,
                        quantiles=None):
    """Compute scaling config for a given bucket."""
    if quantiles is None:
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    bmin, bmax = SIZE_BUCKETS[bucket]
    challenge = {
        "task": {"task_params": {
            "context_len": 512, "prediction_len": 96,
            "num_variates": num_variates, "quantiles": quantiles,
        }},
        "flops_budget": {"min": bmin, "max": bmax},
    }
    return _compute_scaling(challenge)


# ── Test helpers ──────────────────────────────────────────────────

class MockClient:
    """Mock GatedClient for testing."""
    def __init__(self, raise_on_call=False):
        self.raise_on_call = raise_on_call
        self.calls = []

    def get_json(self, url):
        self.calls.append(("GET", url))
        if self.raise_on_call:
            raise RuntimeError("mock error")
        return {}

    def post_json(self, url, payload):
        self.calls.append(("POST", url, payload))
        if self.raise_on_call:
            raise RuntimeError("mock error")
        return {}


def _make_challenge(bucket="small"):
    flops_min, flops_max = SIZE_BUCKETS.get(bucket, (500_000, 2_000_000))
    return {
        "min_flops_equivalent": flops_min,
        "max_flops_equivalent": flops_max,
        "feasible_frontier": [],
        "task": {"task_params": {
            "context_len": 512, "prediction_len": 96,
            "num_variates": 1,
            "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }},
        "db_url": "",
        "llm_url": "",
        "seed": 42,
        "round_id": 1,
    }


def _run(challenge, client):
    """Run design_architecture with injected globals."""
    import builtins
    _mod.load_scratchpad = lambda c: None
    _mod.save_scratchpad = lambda c, d: None
    builtins.load_scratchpad = lambda c: None
    builtins.save_scratchpad = lambda c, d: None
    try:
        return design_architecture(challenge, client)
    finally:
        del builtins.load_scratchpad
        del builtins.save_scratchpad


# ══════════════════════════════════════════════════════════════════
#  1. All generated code passes AST validation
# ══════════════════════════════════════════════════════════════════

class TestCodeValidity:
    def test_all_buckets_produce_valid_code(self):
        """Every scaling config produces code that passes validate_code."""
        for bucket in SIZE_BUCKETS:
            cfg = _scaling_for_bucket(bucket)
            code = _generate_code(cfg)
            ok, errors = validate_code(code)
            assert ok, f"Bucket '{bucket}' failed validation: {errors}"

    def test_code_has_required_functions(self):
        for bucket in SIZE_BUCKETS:
            cfg = _scaling_for_bucket(bucket)
            code = _generate_code(cfg)
            assert "def build_model" in code, f"{bucket} missing build_model"
            assert "def build_optimizer" in code, f"{bucket} missing build_optimizer"

    def test_code_has_optional_hooks(self):
        for bucket in SIZE_BUCKETS:
            cfg = _scaling_for_bucket(bucket)
            code = _generate_code(cfg)
            assert "def training_config" in code, f"{bucket} missing training_config"
            assert "def build_scheduler" in code, f"{bucket} missing build_scheduler"
            assert "def init_weights" in code, f"{bucket} missing init_weights"

    def test_code_has_revin(self):
        for bucket in SIZE_BUCKETS:
            cfg = _scaling_for_bucket(bucket)
            code = _generate_code(cfg)
            assert "RevIN" in code, f"{bucket} missing RevIN"

    def test_code_has_patch_decoder(self):
        for bucket in SIZE_BUCKETS:
            cfg = _scaling_for_bucket(bucket)
            code = _generate_code(cfg)
            assert "PatchDecoder" in code, f"{bucket} missing PatchDecoder"

    def test_code_has_mlp_block(self):
        for bucket in SIZE_BUCKETS:
            cfg = _scaling_for_bucket(bucket)
            code = _generate_code(cfg)
            assert "MLPBlock" in code, f"{bucket} missing MLPBlock"


# ══════════════════════════════════════════════════════════════════
#  2. Output shapes (if torch available)
# ══════════════════════════════════════════════════════════════════

class TestOutputShapes:
    def _check_shape(self, bucket):
        cfg = _scaling_for_bucket(bucket)
        code = _generate_code(cfg)
        try:
            import torch
        except ImportError:
            return
        ns = {}
        exec(compile(code, f"<{bucket}_patch_decoder>", "exec"), ns)
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        model = ns["build_model"](512, 96, 1, quantiles)
        x = torch.randn(2, 512, 1)
        out = model(x)
        assert out.shape == (2, 96, 1, 9), \
            f"Bucket '{bucket}' shape {out.shape} != (2, 96, 1, 9)"

    def test_tiny_shape(self):
        self._check_shape("tiny")

    def test_small_shape(self):
        self._check_shape("small")

    def test_medium_small_shape(self):
        self._check_shape("medium_small")

    def test_medium_shape(self):
        self._check_shape("medium")

    def test_large_shape(self):
        self._check_shape("large")


# ══════════════════════════════════════════════════════════════════
#  3. Scaling correctness
# ══════════════════════════════════════════════════════════════════

class TestScaling:
    def test_all_buckets_produce_valid_scaling(self):
        for bucket in SIZE_BUCKETS:
            cfg = _scaling_for_bucket(bucket)
            assert cfg["d_model"] >= 4, f"d_model too small for {bucket}: {cfg['d_model']}"
            assert cfg["n_layers"] >= 1, f"n_layers too small for {bucket}"
            assert cfg["patch_size"] >= 4, f"patch_size too small for {bucket}"

    def test_larger_budgets_allow_larger_models(self):
        """Larger FLOPs budgets should produce configs with more total parameters."""
        order = ["tiny", "small", "medium_small", "medium", "large"]
        params = []
        for b in order:
            cfg = _scaling_for_bucket(b)
            # Rough parameter count proxy: d_model^2 * n_layers
            p = cfg["d_model"] ** 2 * cfg["n_layers"]
            params.append(p)
        for i in range(len(params) - 1):
            assert params[i] <= params[i + 1], \
                f"Model capacity should increase: {order[i]}={params[i]} > {order[i+1]}={params[i+1]}"


# ══════════════════════════════════════════════════════════════════
#  4. Determinism — same input always gives same output
# ══════════════════════════════════════════════════════════════════

class TestDeterminism:
    def test_same_challenge_same_code(self):
        """Calling _generate_code twice with same config gives identical code."""
        for bucket in SIZE_BUCKETS:
            cfg = _scaling_for_bucket(bucket)
            code1 = _generate_code(cfg)
            code2 = _generate_code(cfg)
            assert code1 == code2, f"Non-deterministic code for {bucket}"

    def test_design_architecture_deterministic(self):
        """design_architecture returns identical code for same challenge."""
        client = MockClient()
        for bucket in SIZE_BUCKETS:
            ch = _make_challenge(bucket)
            r1 = _run(ch, client)
            r2 = _run(ch, client)
            assert r1["code"] == r2["code"], f"Non-deterministic for {bucket}"
            assert r1["name"] == r2["name"]


# ══════════════════════════════════════════════════════════════════
#  5. Never-empty guarantee
# ══════════════════════════════════════════════════════════════════

class TestNeverEmpty:
    def test_all_buckets_return_code(self):
        client = MockClient()
        for bucket in SIZE_BUCKETS:
            ch = _make_challenge(bucket)
            result = _run(ch, client)
            assert result["code"], f"Empty code for {bucket}"
            ok, errors = validate_code(result["code"])
            assert ok, f"Invalid code for {bucket}: {errors}"

    def test_client_errors_still_produce_code(self):
        """Even if client raises, deterministic agent doesn't need it."""
        client = MockClient(raise_on_call=True)
        ch = _make_challenge("medium")
        result = _run(ch, client)
        assert result["code"]
        ok, _ = validate_code(result["code"])
        assert ok

    def test_no_llm_calls(self):
        """Deterministic agent should never call the client."""
        client = MockClient()
        ch = _make_challenge("small")
        _run(ch, client)
        assert len(client.calls) == 0, f"Unexpected client calls: {client.calls}"


# ══════════════════════════════════════════════════════════════════
#  6. Challenge parsing
# ══════════════════════════════════════════════════════════════════

class TestChallengeParsing:
    def test_flat_format(self):
        ch = {"min_flops_equivalent": 500_000, "max_flops_equivalent": 2_000_000}
        fmin, fmax = extract_flops_budget(ch)
        assert fmin == 500_000
        assert fmax == 2_000_000

    def test_nested_format(self):
        ch = {"flops_budget": {"min": 100_000, "max": 500_000}}
        fmin, fmax = extract_flops_budget(ch)
        assert fmin == 100_000
        assert fmax == 500_000

    def test_target_flops_in_motivation(self):
        """The agent should include its target FLOPs in motivation."""
        client = MockClient()
        ch = _make_challenge("medium")
        result = _run(ch, client)
        # medium bucket: (10M, 50M), target = 60% of 50M = 30M
        assert "30,000,000" in result["motivation"]
