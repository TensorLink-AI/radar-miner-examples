"""Tests for score-feedback merge into history state.

Covers ``merge_results_into_state``, ``best_own_submission``, and the
score-aware ``format_history`` rendering. Validator-supplied
``previous_results`` are joined onto matching history entries by
``code_hash``.
"""

from __future__ import annotations

import os
import sys

# These helpers currently live in the openai_sdk agent's core/. Use
# that copy explicitly — the autonomous core/ does not yet have them.
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "agents", "openai_sdk"),
)
for _k in list(sys.modules.keys()):
    if _k == "core" or _k.startswith("core."):
        del sys.modules[_k]

from core.history import (  # noqa: E402
    add_entry,
    best_own_submission,
    format_history,
    merge_results_into_state,
)


def _entry(name: str, code: str, **kw) -> dict:
    state = add_entry({}, name=name, code=code, motivation="m", **kw)
    return state["history"][0]


class TestMergeResultsIntoState:
    def test_missing_results_no_op(self):
        state = {"history": [_entry("a", "code-a")]}
        merge_results_into_state(state, None)
        assert "score" not in state["history"][0]
        merge_results_into_state(state, [])
        assert "score" not in state["history"][0]

    def test_missing_score_key_skipped(self):
        h = _entry("a", "code-a")
        state = {"history": [h]}
        merge_results_into_state(state, [
            {"code_hash": h["code_hash"], "rank": 2, "rank_total": 5},
        ])
        assert state["history"][0]["rank"] == 2
        assert state["history"][0]["rank_total"] == 5
        assert "score" not in state["history"][0]

    def test_score_merged_by_code_hash(self):
        h = _entry("a", "code-a")
        state = {"history": [h]}
        merge_results_into_state(state, [
            {
                "round_id": "r1",
                "code_hash": h["code_hash"],
                "score": 0.337,
                "rank": 3,
                "rank_total": 12,
                "error": None,
            },
        ])
        e = state["history"][0]
        assert e["score"] == 0.337
        assert e["rank"] == 3
        assert e["rank_total"] == 12
        assert e["scored_round_id"] == "r1"
        assert "error" not in e

    def test_error_recorded(self):
        h = _entry("a", "code-a")
        state = {"history": [h]}
        merge_results_into_state(state, [
            {"code_hash": h["code_hash"], "score": None,
             "error": "trainer OOM"},
        ])
        assert state["history"][0]["error"] == "trainer OOM"

    def test_unknown_hash_ignored(self):
        h = _entry("a", "code-a")
        state = {"history": [h]}
        merge_results_into_state(state, [
            {"code_hash": 999999, "score": 0.5},
        ])
        assert "score" not in state["history"][0]

    def test_duplicate_code_hashes_all_updated(self):
        # Same code submitted across two rounds — both entries get the
        # score so the agent doesn't have to dedupe.
        h1 = _entry("a", "same-code", bucket="small")
        h2 = _entry("b", "same-code", bucket="small")
        assert h1["code_hash"] == h2["code_hash"]
        state = {"history": [h1, h2]}
        merge_results_into_state(state, [
            {"code_hash": h1["code_hash"], "score": 0.42},
        ])
        assert state["history"][0]["score"] == 0.42
        assert state["history"][1]["score"] == 0.42

    def test_handles_missing_history_key(self):
        state = {}
        merge_results_into_state(state, [
            {"code_hash": 1, "score": 0.5},
        ])
        # Creates the list, but no entries match → no-op contents
        assert state["history"] == []


class TestFormatHistoryWithScores:
    def test_minimize_sorts_lowest_first(self):
        entries = [
            {"name": "a", "score": 0.5, "bucket": "s",
             "strategy": "x", "motivation": "alpha"},
            {"name": "b", "score": 0.2, "bucket": "s",
             "strategy": "x", "motivation": "beta"},
            {"name": "c", "score": 0.8, "bucket": "s",
             "strategy": "x", "motivation": "gamma"},
        ]
        out = format_history(entries, score_direction="minimize")
        lines = [ln for ln in out.split("\n") if ln.strip()]
        assert lines[0].startswith("- b ")
        assert lines[1].startswith("- a ")
        assert lines[2].startswith("- c ")

    def test_maximize_sorts_highest_first(self):
        entries = [
            {"name": "a", "score": 0.5, "bucket": "s",
             "strategy": "x", "motivation": ""},
            {"name": "b", "score": 0.9, "bucket": "s",
             "strategy": "x", "motivation": ""},
        ]
        out = format_history(entries, score_direction="maximize")
        lines = [ln for ln in out.split("\n") if ln.strip()]
        assert lines[0].startswith("- b ")
        assert lines[1].startswith("- a ")

    def test_pending_marked_and_after_scored(self):
        entries = [
            {"name": "pend", "bucket": "s", "strategy": "x",
             "motivation": ""},
            {"name": "done", "score": 0.1, "bucket": "s",
             "strategy": "x", "motivation": ""},
        ]
        out = format_history(entries)
        lines = [ln for ln in out.split("\n") if ln.strip()]
        assert lines[0].startswith("- done ")
        assert "score=" in lines[0]
        assert lines[1].startswith("- pend ")
        assert "(pending)" in lines[1]

    def test_score_with_rank(self):
        entries = [
            {"name": "a", "score": 0.337, "rank": 3, "rank_total": 12,
             "bucket": "s", "strategy": "x", "motivation": "ok"},
        ]
        out = format_history(entries)
        assert "score=0.337" in out
        assert "rank=3/12" in out

    def test_round_trip_through_merge(self):
        h = _entry("a", "code-a", bucket="small", strategy="seed")
        state = {"history": [h]}
        merge_results_into_state(state, [
            {"code_hash": h["code_hash"], "score": 0.3, "rank": 3,
             "rank_total": 12},
        ])
        out = format_history(state["history"])
        assert "score=0.3" in out
        assert "rank=3/12" in out


class TestBestOwnSubmission:
    def test_none_when_no_scored(self):
        state = {"history": [_entry("a", "code-a")]}
        assert best_own_submission(state) is None

    def test_minimize_picks_lowest(self):
        state = {"history": [
            {"name": "a", "score": 0.5, "code_hash": 1},
            {"name": "b", "score": 0.2, "code_hash": 2},
            {"name": "c", "score": 0.8, "code_hash": 3},
        ]}
        assert best_own_submission(state, "minimize")["name"] == "b"

    def test_maximize_picks_highest(self):
        state = {"history": [
            {"name": "a", "score": 0.5, "code_hash": 1},
            {"name": "b", "score": 0.9, "code_hash": 2},
        ]}
        assert best_own_submission(state, "maximize")["name"] == "b"

    def test_ignores_unscored_entries(self):
        state = {"history": [
            {"name": "pending", "code_hash": 1},
            {"name": "scored", "score": 0.42, "code_hash": 2},
        ]}
        assert best_own_submission(state)["name"] == "scored"
