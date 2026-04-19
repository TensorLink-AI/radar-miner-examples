"""Tests for the geometric-probe-then-refine FLOP sizing helper.

The sweep itself is task-agnostic — it operates on a ``probe(size) ->
flops`` callback. These tests use cheap arithmetic probes (linear,
piecewise, noisy) so they run in microseconds and don't depend on
PyTorch model construction.
"""

from __future__ import annotations

import os
import sys

# ``core.sizing`` only lives in the openai_sdk agent's core/. Import it
# explicitly — the autonomous agent's core/ does not have it.
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "agents", "openai_sdk"),
)
for _k in list(sys.modules.keys()):
    if _k == "core" or _k.startswith("core."):
        del sys.modules[_k]

from core.sizing import (  # noqa: E402
    MAX_PROBES,
    N_PROBE_GEOMETRIC,
    N_PROBE_REFINE,
    sweep_sizes,
)


def _linear(scale: int = 100):
    def probe(size: int) -> int:
        return size * scale
    return probe


def _stepwise(scale: int = 100, step: int = 32):
    """FLOPs flat across each ``step``-wide band — mimics head-count
    rounding where several adjacent knob values produce identical FLOPs.
    Binary search oscillates on these; geometric+refine doesn't.
    """
    def probe(size: int) -> int:
        bucket = (size // step) * step
        return bucket * scale
    return probe


class TestSweepBasics:
    def test_finds_close_to_target_on_monotonic_probe(self):
        out = sweep_sizes(_linear(100), 1, 10_000,
                          target_flops=500_000)
        assert out["best"] is not None
        size, flops = out["best"]
        # target is 500K; linear probe gives size*100. Exact solution is
        # size=5000. Geometric+refine should land within 10 percent.
        assert abs(flops - 500_000) <= 50_000

    def test_probes_are_measured_not_interpolated(self):
        seen: list[int] = []

        def probe(size: int) -> int:
            seen.append(size)
            return size * 100

        out = sweep_sizes(probe, 1, 10_000, target_flops=500_000)
        size, _ = out["best"]
        assert size in seen, "best size must be a size we actually probed"

    def test_returns_all_successful_probes_sorted(self):
        out = sweep_sizes(_linear(10), 1, 1000, target_flops=5000)
        sizes = [s for s, _ in out["probes"]]
        assert sizes == sorted(sizes)

    def test_single_size_range(self):
        out = sweep_sizes(_linear(100), 100, 100, target_flops=10_000)
        assert out["best"] is not None
        assert out["best"][0] == 100


class TestProbeFailures:
    def test_exception_treated_as_unreachable(self):
        def probe(size: int) -> int:
            if size < 100:
                raise RuntimeError("too small")
            return size * 10

        out = sweep_sizes(probe, 1, 10_000, target_flops=5000)
        assert out["best"] is not None
        assert all(s >= 100 for s, _ in out["probes"])
        assert any(s < 100 for s in out["failures"])

    def test_none_return_treated_as_unreachable(self):
        def probe(size: int):
            return None if size < 200 else size * 10

        out = sweep_sizes(probe, 1, 10_000, target_flops=5000)
        assert all(s >= 200 for s, _ in out["probes"])
        assert any(s < 200 for s in out["failures"])

    def test_zero_return_treated_as_unreachable(self):
        def probe(size: int) -> int:
            return 0 if size < 500 else size * 10

        out = sweep_sizes(probe, 1, 10_000, target_flops=5000)
        assert all(s >= 500 for s, _ in out["probes"])

    def test_all_probes_fail_returns_none_best(self):
        def probe(size: int) -> int:
            raise ValueError("always")

        out = sweep_sizes(probe, 1, 1000, target_flops=100)
        assert out["best"] is None
        assert out["probes"] == []
        assert len(out["failures"]) >= 1


class TestHardCap:
    def test_never_exceeds_max_probes(self):
        calls: list[int] = []

        def probe(size: int) -> int:
            calls.append(size)
            return size * 10

        sweep_sizes(probe, 1, 1_000_000, target_flops=5000)
        assert len(calls) <= MAX_PROBES
        # Geometric + at least one refine pass means we shouldn't be
        # leaving half the probe budget on the table when the range is
        # wide enough.
        assert len(calls) >= N_PROBE_GEOMETRIC

    def test_refine_phase_runs_within_cap(self):
        calls: list[int] = []

        def probe(size: int) -> int:
            calls.append(size)
            return size * 10

        sweep_sizes(probe, 1, 10_000_000, target_flops=5_000_000)
        assert len(calls) <= N_PROBE_GEOMETRIC + N_PROBE_REFINE


class TestOutOfRangeTargets:
    def test_target_below_all_probes(self):
        # Linear *1_000_000 means every size >= 100 exceeds target=100.
        out = sweep_sizes(_linear(1_000_000), 100, 10_000,
                          target_flops=100)
        assert out["best"] is not None
        size, flops = out["best"]
        # The minimum achievable flops is size_min * 1M = 100M.
        # Best is the smallest successful probe.
        assert size == 100
        assert flops >= 100

    def test_target_above_all_probes(self):
        out = sweep_sizes(_linear(10), 1, 100,
                          target_flops=5_000_000_000)
        assert out["best"] is not None
        size, flops = out["best"]
        # Best is the largest achievable.
        assert size == 100
        assert flops == 1000


class TestInvalidInputs:
    def test_reversed_range_returns_empty(self):
        out = sweep_sizes(_linear(100), 100, 10, target_flops=1000)
        assert out["best"] is None
        assert out["probes"] == []
        assert out["n_probes"] == 0

    def test_zero_size_min_returns_empty(self):
        out = sweep_sizes(_linear(100), 0, 1000, target_flops=1000)
        assert out["best"] is None
        assert out["n_probes"] == 0

    def test_negative_size_min_returns_empty(self):
        out = sweep_sizes(_linear(100), -10, 1000, target_flops=1000)
        assert out["best"] is None


class TestPiecewiseMonotonic:
    """Real FLOPs are monotonic but piecewise (head-count divisibility,
    group counts, kernel floors). Binary search oscillates on flat
    regions; geometric+refine handles them without fuss."""

    def test_flat_regions_do_not_stall_sweep(self):
        out = sweep_sizes(_stepwise(100, 32), 1, 10_000,
                          target_flops=500_000)
        assert out["best"] is not None
        # Successful probes should be strictly nondecreasing by size.
        probes = out["probes"]
        for (_, f1), (_, f2) in zip(probes, probes[1:]):
            assert f1 <= f2

    def test_target_inside_flat_band_returns_measured_point(self):
        # Target = 5000. Stepwise(scale=100, step=32) -> flops =
        # (size // 32) * 32 * 100. The band giving 5000 is size 16..31
        # (since (31 // 32)*32*100 = 0 — ok, let me pick target 9600).
        # Band (300..331) maps to (300//32)*32*100 = 9*32*100 = 28800.
        # Simpler: target 3200 (flops == 3200 for size 32..63).
        out = sweep_sizes(_stepwise(100, 32), 1, 1000,
                          target_flops=3200)
        assert out["best"] is not None
        # Best MEASURED flops should be close to target (within the
        # band or just outside).
        _, best_flops = out["best"]
        assert abs(best_flops - 3200) <= 32 * 100
