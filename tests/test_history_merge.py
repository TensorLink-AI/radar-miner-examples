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
    NOTES_MAX_ENTRIES,
    NOTES_SECTIONS,
    add_entry,
    add_note,
    best_own_submission,
    format_history,
    format_notes,
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


class TestAddNote:
    def test_sections_constant_matches_expectation(self):
        # Locking in the three-section contract the prompt advertises.
        assert set(NOTES_SECTIONS) == {
            "open_hypotheses", "dead_ends", "task_observations",
        }

    def test_appends_to_hypothesis_section(self):
        state: dict = {}
        add_note(state, "open_hypotheses", "try depthwise-sep convs")
        assert state["notes"]["open_hypotheses"] == [
            "try depthwise-sep convs",
        ]

    def test_appends_to_dead_ends_section(self):
        state: dict = {}
        add_note(state, "dead_ends", "plain MLP — undersized")
        assert state["notes"]["dead_ends"] == ["plain MLP — undersized"]

    def test_appends_to_task_observations_section(self):
        state: dict = {}
        add_note(state, "task_observations", "output shape is (B, N, K)")
        assert state["notes"]["task_observations"] == [
            "output shape is (B, N, K)",
        ]

    def test_strips_whitespace(self):
        state: dict = {}
        add_note(state, "open_hypotheses", "  idea one  \n")
        assert state["notes"]["open_hypotheses"] == ["idea one"]

    def test_rejects_unknown_section(self):
        import pytest
        with pytest.raises(ValueError):
            add_note({}, "bogus_section", "anything")

    def test_empty_string_is_noop(self):
        state: dict = {}
        add_note(state, "open_hypotheses", "")
        add_note(state, "open_hypotheses", "   ")
        # add_note initialises the sections lazily; either empty dict or
        # empty list is an acceptable "nothing was written" shape.
        assert state.get("notes", {}).get("open_hypotheses", []) == []

    def test_non_string_is_noop(self):
        state: dict = {}
        add_note(state, "open_hypotheses", None)  # type: ignore[arg-type]
        add_note(state, "open_hypotheses", 42)    # type: ignore[arg-type]
        assert state.get("notes", {}).get("open_hypotheses", []) == []

    def test_cap_drops_oldest(self):
        state: dict = {}
        for i in range(NOTES_MAX_ENTRIES + 5):
            add_note(state, "dead_ends", f"attempt_{i}")
        bucket = state["notes"]["dead_ends"]
        assert len(bucket) == NOTES_MAX_ENTRIES
        # Oldest 5 got dropped; newest entry is attempt_{N+4}
        assert bucket[0] == "attempt_5"
        assert bucket[-1] == f"attempt_{NOTES_MAX_ENTRIES + 4}"

    def test_cap_is_per_section(self):
        state: dict = {}
        for i in range(NOTES_MAX_ENTRIES):
            add_note(state, "open_hypotheses", f"h{i}")
        # Filling hypotheses to the cap must not evict anything from
        # another section.
        add_note(state, "dead_ends", "one dead end")
        add_note(state, "task_observations", "one observation")
        notes = state["notes"]
        assert len(notes["open_hypotheses"]) == NOTES_MAX_ENTRIES
        assert notes["dead_ends"] == ["one dead end"]
        assert notes["task_observations"] == ["one observation"]


class TestFormatNotes:
    def test_empty_state_renders_empty_string(self):
        assert format_notes({}) == ""
        assert format_notes({"notes": {}}) == ""

    def test_renders_all_three_sections(self):
        state: dict = {}
        add_note(state, "open_hypotheses", "hypothesis one")
        add_note(state, "dead_ends", "dead end one")
        add_note(state, "task_observations", "observation one")
        out = format_notes(state)
        assert "## Open Hypotheses" in out
        assert "## Dead Ends" in out
        assert "## Task Observations" in out
        assert "- hypothesis one" in out
        assert "- dead end one" in out
        assert "- observation one" in out

    def test_skips_empty_sections(self):
        state: dict = {}
        add_note(state, "dead_ends", "only this")
        out = format_notes(state)
        assert "## Dead Ends" in out
        assert "Open Hypotheses" not in out
        assert "Task Observations" not in out

    def test_stable_section_ordering(self):
        state: dict = {}
        # Write in a different order than NOTES_SECTIONS to prove the
        # render respects the canonical order, not insertion order.
        add_note(state, "task_observations", "obs")
        add_note(state, "open_hypotheses", "hyp")
        add_note(state, "dead_ends", "dead")
        out = format_notes(state)
        hyp_idx = out.index("Open Hypotheses")
        dead_idx = out.index("Dead Ends")
        obs_idx = out.index("Task Observations")
        assert hyp_idx < dead_idx < obs_idx
