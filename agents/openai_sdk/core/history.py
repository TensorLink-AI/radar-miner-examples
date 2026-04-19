"""Persistent experiment history across rounds — stored in scratchpad state."""

import json
import os
import time


SIZE_BUCKETS = {
    "tiny":         (100_000,       500_000),
    "small":        (500_000,     2_000_000),
    "medium_small": (2_000_000,  10_000_000),
    "medium":       (10_000_000, 50_000_000),
    "large":        (50_000_000, 125_000_000),
}


def identify_bucket(flops_min: int, flops_max: int) -> str:
    """Identify which size bucket matches the given FLOPs range."""
    for name, (bmin, bmax) in SIZE_BUCKETS.items():
        if bmin == flops_min and bmax == flops_max:
            return name
    # Fuzzy match: find closest bucket by midpoint
    mid = (flops_min + flops_max) / 2
    best = "medium"
    best_dist = float("inf")
    for name, (bmin, bmax) in SIZE_BUCKETS.items():
        bmid = (bmin + bmax) / 2
        dist = abs(mid - bmid)
        if dist < best_dist:
            best_dist = dist
            best = name
    return best


def extract_flops_budget(challenge: dict) -> tuple[int, int]:
    """Extract FLOPs min/max from challenge, supporting both field formats."""
    fb = challenge.get("flops_budget", {})
    if isinstance(fb, dict) and (fb.get("min") or fb.get("max")):
        return int(fb.get("min", 0)), int(fb.get("max", 0))
    fmin = challenge.get("min_flops_equivalent", 0)
    fmax = challenge.get("max_flops_equivalent", 0)
    if fmin or fmax:
        return int(fmin), int(fmax)
    return 0, 0


def get_history(state: dict) -> list[dict]:
    """Get experiment history from scratchpad state."""
    return state.get("history", [])


def get_bucket_history(state: dict, bucket: str) -> list[dict]:
    """Get history entries for a specific size bucket."""
    return [e for e in get_history(state) if e.get("bucket") == bucket]


def format_history(entries: list[dict], max_entries: int = 10,
                   score_direction: str = "minimize") -> str:
    """Format history entries for inclusion in prompts.

    Scored entries come first (best per ``score_direction``), unscored
    entries follow (newest first), marked ``(pending)``. Output is capped
    at ``max_entries`` lines.
    """
    if not entries:
        return "No previous submissions."

    def _has_score(e: dict) -> bool:
        return isinstance(e.get("score"), (int, float))

    scored = [e for e in entries if _has_score(e)]
    pending = [e for e in entries if not _has_score(e)]
    reverse = score_direction == "maximize"
    scored.sort(key=lambda e: e["score"], reverse=reverse)
    pending.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
    ordered = scored + pending

    lines: list[str] = []
    for e in ordered[:max_entries]:
        bits = [
            f"bucket={e.get('bucket', '?')}",
            f"strategy={e.get('strategy', '?')}",
        ]
        if _has_score(e):
            score_bit = f"score={e['score']:.4g}"
            rank = e.get("rank")
            total = e.get("rank_total")
            if isinstance(rank, int):
                score_bit += (
                    f", rank={rank}/{total}" if isinstance(total, int)
                    else f", rank={rank}"
                )
            bits.append(score_bit)
            err = e.get("error")
            if err:
                bits.append(f"error={str(err)[:40]!r}")
        else:
            bits.append("(pending)")
        lines.append(
            f"- {e.get('name', '?')} ({', '.join(bits)}): "
            f"{e.get('motivation', '')}"
        )
    return "\n".join(lines)


def merge_results_into_state(state: dict,
                             previous_results: list[dict] | None) -> dict:
    """Merge validator-supplied round results onto matching history entries.

    Expected shape (harness-provided, optional)::

        challenge["previous_results"] = [
            {"round_id": str, "code_hash": int, "score": float | None,
             "rank": int | None, "error": str | None},
            ...
        ]

    Join key is ``code_hash``. Any entries in ``state["history"]`` whose
    ``code_hash`` matches get ``score``, ``rank``, ``rank_total``,
    ``error``, and ``scored_round_id`` written onto them. Duplicate
    hashes all receive the update. Unknown hashes are ignored. Returns
    the mutated state for chaining.
    """
    if not previous_results:
        return state
    history = state.setdefault("history", [])
    index: dict = {}
    for entry in history:
        h = entry.get("code_hash")
        if h is not None:
            index.setdefault(h, []).append(entry)
    for r in previous_results:
        if not isinstance(r, dict):
            continue
        h = r.get("code_hash")
        if h is None:
            continue
        targets = index.get(h) or []
        if not targets:
            continue
        for entry in targets:
            score = r.get("score")
            if isinstance(score, (int, float)):
                entry["score"] = float(score)
            rank = r.get("rank")
            if isinstance(rank, int):
                entry["rank"] = rank
            total = r.get("rank_total")
            if isinstance(total, int):
                entry["rank_total"] = total
            err = r.get("error")
            if err:
                entry["error"] = str(err)
            rid = r.get("round_id")
            if rid is not None:
                entry["scored_round_id"] = rid
    return state


def best_own_submission(state: dict,
                        score_direction: str = "minimize") -> dict | None:
    """Return the best-scored history entry, or ``None`` if none are scored.

    ``score_direction`` is ``"minimize"`` (lower is better) or ``"maximize"``.
    Ties break by most-recent timestamp.
    """
    scored = [
        e for e in state.get("history", [])
        if isinstance(e.get("score"), (int, float))
    ]
    if not scored:
        return None
    if score_direction == "maximize":
        scored.sort(key=lambda e: (-e["score"], -e.get("timestamp", 0)))
    else:
        scored.sort(key=lambda e: (e["score"], -e.get("timestamp", 0)))
    return scored[0]


def add_entry(state: dict, *, name: str, code: str, motivation: str,
              bucket: str = "", flops: int = 0,
              strategy: str = "", metadata: dict | None = None) -> dict:
    """Add a history entry to the scratchpad state. Returns updated state."""
    if "history" not in state:
        state["history"] = []
    entry = {
        "name": name,
        "motivation": motivation,
        "code_hash": hash(code) & 0xFFFFFFFF,
        "code_length": len(code),
        "bucket": bucket,
        "flops_target": flops,
        "strategy": strategy,
        "timestamp": time.time(),
    }
    if metadata:
        entry["metadata"] = metadata
    state["history"].append(entry)

    # Keep last 50 entries to avoid scratchpad bloat
    if len(state["history"]) > 50:
        state["history"] = state["history"][-50:]

    return state


def load_state(scratch_dir: str) -> dict:
    """Load state dict from scratch_dir/state.json."""
    path = os.path.join(scratch_dir, "state.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_state(scratch_dir: str, state: dict) -> None:
    """Save state dict to scratch_dir/state.json."""
    os.makedirs(scratch_dir, exist_ok=True)
    path = os.path.join(scratch_dir, "state.json")
    with open(path, "w") as f:
        json.dump(state, f, indent=2)
