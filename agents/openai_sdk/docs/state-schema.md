# openai_sdk Agent — Extended State Schema Design

Planning doc for four features that share the scratchpad-persisted state
object managed by `core/history.py`:

1. Code lineage / candidate IDs
2. `read_my_submissions` tool
3. Structured scratchpad readback (hypothesis → candidate → outcome)
4. Tool macros

The features ship in order, but they extend one schema, so the schema is
designed up front. No code yet — this doc is for review before
implementation of feature 1.

---

## 1. Current state schema

Everything below lives in `state.json` inside the scratchpad directory.
Loaded by `history.load_state` and saved by `history.save_state`. The
in-memory wrapper `state_holder` (in `tools.py:831`) holds the dict plus
two round-scoped flags; only the `state` sub-dict is persisted.

### 1.1 Persisted keys

| Key            | Type             | Owner                       | Notes |
|----------------|------------------|-----------------------------|-------|
| `history`      | `list[dict]`     | `add_entry`, `merge_results_into_state` | Submission records. Capped at 50 (drops oldest). |
| `notes`        | `dict[str, list[str]]` | `add_note`, `format_notes` | Three sub-lists: `open_hypotheses`, `dead_ends`, `task_observations`. Each capped at `NOTES_MAX_ENTRIES = 20`. |
| `agent_notes`  | `str` (optional) | deprecated `write_scratchpad(notes=...)` | Free-form string; only present on legacy scratchpads. |

#### History entry shape (current)

Written by `add_entry` in `core/history.py:180`:

```python
{
    "name": str,
    "motivation": str,
    "code_hash": int,            # hash(code) & 0xFFFFFFFF — join key for previous_results
    "code_length": int,
    "bucket": str,               # tiny | small | medium_small | medium | large
    "flops_target": int,
    "strategy": str,
    "timestamp": float,
    "metadata": dict,            # optional, free-form
    # ── added by merge_results_into_state when previous_results arrives ──
    "score": float,              # optional
    "rank": int,                 # optional
    "rank_total": int,           # optional
    "error": str,                # optional
    "scored_round_id": str,      # optional
}
```

Note: the **code itself is not stored** — only `code_hash`. That's a
deliberate choice for scratchpad size. Today this means we cannot
reconstruct what was actually submitted from history alone.

### 1.2 Round-scoped (`state_holder`, not persisted)

| Key                    | Type   | Set by                         | Read by |
|------------------------|--------|--------------------------------|---------|
| `state`                | `dict` | `build_handlers`               | every state-touching handler |
| `wrote_this_round`     | `bool` | `_write_scratchpad`            | `_submit` (warns if false) |
| `last_validated_code`  | `str`  | `_validate_code` on success    | `agent.py` auto-submit recovery |

---

## 2. Proposed extended schema

Three new top-level persisted keys: `candidates`, `hypotheses`, `macros`.
The existing `history` entries gain one new field (`candidate_id`). The
existing `notes` block stays as-is for backwards compatibility but the
LLM is steered toward the structured `hypotheses` list instead.

### 2.1 New top-level keys

| Key            | Type                       | Used by features |
|----------------|----------------------------|------------------|
| `candidates`   | `dict[str, CandidateRecord]` (keyed by `cand_<8 hex>`) | 1, 3 (link target) |
| `hypotheses`   | `list[HypothesisRecord]`   | 3 |
| `macros`       | `dict[str, MacroRecord]`   | 4 |

### 2.2 `CandidateRecord` — feature 1 (as shipped)

A candidate is a piece of code the agent has worked on. The
``candidates`` state key is a **dict keyed by id**, where the id is a
stable hash of the code itself (``cand_<8 hex>``). Identical code
yields the same id, so re-sketching or re-validating the same source
naturally deduplicates instead of producing parallel records.

```python
state["candidates"] = {
    "cand_a3f24c1d": {
        "code": str,                # the full source
        "flops": int | None,        # measured by sketch_architecture
        "trace": str | None,        # rendered per-layer trace, or None
        "validated": bool,          # set True by validate_code on success
        "submitted": bool,          # set True by submit when this id ships
        "created_at": float,        # time.time() at first insert
    },
    ...
}
```

**Writes:**
- `_sketch_architecture` calls `upsert_candidate` with the code, FLOPs,
  and trace. Returns the id and surfaces it as the last line of the
  tool output (`candidate_id: cand_a3f24c1d`).
- `_validate_code` either looks up an existing candidate by id (and
  ignores any `code` arg) or, if no id was passed and validation
  passes, calls `upsert_candidate` to create one. On success it sets
  `validated=True`.
- `_submit` looks up the candidate by id and ships its stored code,
  setting `submitted=True`. Without an id, submit behaves as it did
  before this feature (uses the `code` arg, no state mutation).

**Reads:** the LLM carries ids forward through tool-result text. Lineage
read tools (`list_candidates`, `get_candidate`) are deferred — the
in-conversation id is enough for feature 1's needs.

**Cap:** none for now. Hash-keyed dedup keeps growth proportional to
unique designs, not call count.

### 2.3 Submission records — feature 2

**No new top-level key.** Submissions already live in `state["history"]`
and are merged with `previous_results` on round entry by
`history.merge_results_into_state` at `agent.py:422`. We just add one
field to the existing entry shape:

```python
{
    # ... existing fields (name, motivation, code_hash, ...) ...
    "candidate_id": str | None,   # NEW — the candidate this entry was submitted from
}
```

**Reads:** new tool `read_my_submissions` walks `state["history"]` and
formats it. Optional flags: `scored_only=True`, `limit=N`,
`include_code=False` (would require us to also store `code` in candidate
records, which is feature 1's job).

The contract is: if you want the **code** of a prior submission, you
follow `history_entry.candidate_id → candidate.code`. History stays
hash-only.

### 2.4 `HypothesisRecord` — feature 3

The current `notes.open_hypotheses` is `list[str]`. We add a new
top-level `hypotheses` list of structured records that link a hypothesis
to a candidate it spawned and an outcome (score) once the round closes.

```python
{
    "id": str,                       # "hyp_0007"
    "round_id": str,                 # round it was authored in
    "created_at": float,
    "text": str,                     # the hypothesis itself
    "candidate_id": str | None,      # candidate produced to test it (set later)
    "outcome": {                     # filled in once previous_results arrives
        "score": float,
        "rank": int | None,
        "scored_round_id": str,
        "verdict": str,              # "supported" | "refuted" | "inconclusive"
    } | None,
}
```

**Writes:**
- `write_scratchpad(hypothesis=..., hypothesis_candidate_id=...)`
  appends a record. If `hypothesis_candidate_id` is omitted the link can
  be filled in later.
- New tool `link_hypothesis(id, candidate_id=..., verdict=...)` for
  late-binding (or for marking "tried it, didn't help").
- On round entry, after `merge_results_into_state` runs, an
  auto-link pass populates `outcome` for any hypothesis whose
  `candidate_id` matches a now-scored history entry.

**Reads:**
- `read_scratchpad` renders open hypotheses with their resolved
  outcomes: e.g. `"hyp_0007: try patchTST patches → cand_0042 → score=0.18 (rank 3/12) [supported]"`.
- The legacy `notes.open_hypotheses` block stays in the rendered output
  as a separate "free-form hypotheses" section so older notes aren't
  hidden, but new hypotheses go into `hypotheses`.

**Cap:** 50 records, drop oldest first. Resolved (outcome != None)
hypotheses are preserved over open ones when capping (i.e. the cap
considers open hypotheses droppable first), so we keep evidence.

### 2.5 `MacroRecord` — feature 4

Macros are reusable tool sequences. Stored in a dict keyed by name for
O(1) lookup.

```python
{
    "name": str,
    "created_at": float,
    "sequence": [
        {"tool": str, "args": dict},   # args may contain "{{var}}" placeholders
        ...
    ],
    "param_names": list[str],          # named placeholders the macro accepts
    "run_count": int,
    "last_run_at": float | None,
    "last_run_round_id": str | None,
}
```

**Writes:** `define_macro(name, sequence, param_names=...)` inserts /
overwrites. `run_macro(name, args)` bumps `run_count` and `last_run_at`.

**Reads:** `list_macros`, `run_macro`. `read_scratchpad` shows a
"Macros: 3 defined" one-liner so the LLM is reminded they exist.

**Caps:**
- `MAX_MACROS = 20`
- `MAX_MACRO_STEPS = 10` per sequence
- `MAX_MACRO_ARGS_BYTES = 8192` for the JSON-serialized args of a single step

### 2.6 Schema-at-a-glance

```python
state = {
    # existing
    "history":          list[HistoryEntry],            # cap 50
    "notes": {
        "open_hypotheses":  list[str],                 # cap 20
        "dead_ends":        list[str],                 # cap 20
        "task_observations": list[str],                # cap 20
    },
    "agent_notes":      str,                           # legacy, optional

    # new
    "candidates":       list[CandidateRecord],         # cap 50
    "hypotheses":       list[HypothesisRecord],        # cap 50
    "macros":           dict[str, MacroRecord],        # cap 20
    "next_candidate_seq": int,                         # monotonic counter
    "next_hypothesis_seq": int,                        # monotonic counter
}
```

---

## 3. Migration

### 3.1 Strategy: lazy default-on-read, no version field

Existing scratchpads on disk lack the new keys. We do **not** introduce
a `schema_version` field. Instead, every accessor uses `.get(key,
default)` or `.setdefault(key, default)` — the same pattern history.py
already follows for `history` and `notes`.

`load_state` itself stays unchanged — it returns whatever JSON is on
disk. Defaults are applied at the helper level:

```python
def get_candidates(state: dict) -> list[dict]:
    return state.get("candidates", [])

def _candidates(state: dict) -> list[dict]:
    return state.setdefault("candidates", [])
```

Same pattern for `hypotheses`, `macros`, `next_candidate_seq` (default
`1`), `next_hypothesis_seq` (default `1`).

### 3.2 Why no auto-upgrade

- A `history` entry from before feature 1 has no `candidate_id`. We
  could fabricate one, but the candidate record wouldn't have any code,
  flops, or sketch results — it'd be a stub. Better to leave
  `candidate_id` absent and treat it as "pre-lineage".
- Old `notes.open_hypotheses` strings could in principle be moved into
  `hypotheses` records, but they have no candidate linkage and no round
  metadata. Same reasoning — leave them where they are; new structured
  hypotheses go into the new list.

### 3.3 Failure mode coverage

If `state.json` predates these changes:
- `read_scratchpad` — sees empty `candidates`/`hypotheses`/`macros`,
  renders the legacy `history` and `notes` blocks normally.
- `read_my_submissions` — works against legacy history, just doesn't
  show `candidate_id` for old entries.
- `list_candidates` / `list_macros` — return empty.
- First `write_scratchpad(hypothesis=...)` after upgrade creates the
  `hypotheses` key on demand and the file is rewritten with the new
  shape on the next `save_state`.

### 3.4 Forward compatibility

If a user manually downgrades the agent to a pre-feature version after
new keys have been written, `save_state` will preserve them as long as
the older code calls `json.dump(state, ...)` over the whole dict, which
it does (`history.py:269`). No data loss on downgrade.

---

## 4. Tool surface changes

### 4.1 Feature 1 — candidate IDs

**Modified tools:**

| Tool | Change |
|---|---|
| `sketch_architecture(code, parent_candidate_id?)` | Adds optional `parent_candidate_id` (string). Result string gains a final line: `candidate_id: cand_0042`. Side effect: writes a `CandidateRecord` with `status="sketched"`. |
| `validate_code(code, candidate_id?)` | Adds optional `candidate_id`. If given, updates the existing candidate's `validate_result` and (on ok) flips `status="validated"`. If omitted, creates a fresh candidate with no parent. Result string gains `candidate_id: cand_0042` line. |
| `submit(code, name, motivation, candidate_id?)` | Adds optional `candidate_id`. Both the new history entry and the submitted candidate get linked. |

**New tools:**

| Tool | Signature | Returns |
|---|---|---|
| `list_candidates()` | no args | JSON array of `{id, status, parent_id, flops_estimated, name}` for the most recent N |
| `get_candidate(candidate_id)` | `candidate_id: str` | Full record incl. code |

### 4.2 Feature 2 — `read_my_submissions`

**New tool:**

```
read_my_submissions(limit?: int = 10, scored_only?: bool = false,
                    include_code?: bool = false) -> str
```

Returns each prior submission as `{name, motivation, candidate_id,
score?, rank?, rank_total?, scored_round_id?, code? (if include_code and
candidate_id resolves)}`. Pulls from `state["history"]`. Code only
appears when `include_code=true` AND the entry's `candidate_id` resolves
to a candidate record we still have.

No modifications to existing tools. The existing `read_scratchpad`
continues to render a summary view; `read_my_submissions` is the
detail view.

### 4.3 Feature 3 — structured scratchpad

**Modified tool — `write_scratchpad`:**

Existing kwargs (`hypothesis`, `dead_end`, `reason`, `observation`,
`notes`) all still work. New kwargs:

| Kwarg | Type | Effect |
|---|---|---|
| `hypothesis_candidate_id` | `str` | When `hypothesis=...` is passed, links the new hypothesis record to this candidate. Ignored if `hypothesis` is empty. |
| `hypothesis_outcome` | `dict` | `{verdict: "supported" \| "refuted" \| "inconclusive"}` — sets a manual verdict without waiting for `previous_results`. Score/rank are filled in later automatically. |

The legacy `notes.open_hypotheses` list still receives the same
hypothesis text (so the old prompt rendering keeps working) AND a
structured record is added to `hypotheses`. We can remove the
double-write later once we're confident nothing reads the legacy list.

**New tool:**

```
link_hypothesis(hypothesis_id: str,
                candidate_id?: str,
                verdict?: "supported" | "refuted" | "inconclusive") -> str
```

Late-bind a candidate or a manual verdict to an existing hypothesis.
Useful when the LLM realizes mid-round that an earlier hypothesis
matches the candidate it just sketched.

**Modified tool — `read_scratchpad`:**

Adds a new section to the rendered output:

```
## Hypotheses
- hyp_0007: try patchTST patches → cand_0042 → score=0.18 (rank 3/12) [supported]
- hyp_0008: drop dropout for tiny bucket → (no candidate yet)
- hyp_0005: bigger context → cand_0033 → score=0.31 (rank 9/12) [refuted]
```

Order: open (no outcome) first, then resolved most-recent-first.

### 4.4 Feature 4 — macros

**New tools:**

```
define_macro(name: str,
             sequence: list[{tool: str, args: dict}],
             param_names?: list[str]) -> str

run_macro(name: str, args?: dict) -> str

list_macros() -> str
```

`run_macro` substitutes `{{var}}` strings inside step args with values
from the `args` dict before invoking each tool. Steps run sequentially
through the same handler dict the LLM uses, so circuit breakers and
counters apply transparently. Failure of any step aborts the macro
(open question §5.6).

---

## 5. Resolved decisions and open questions

### 5.0 Resolved (locked in for feature 1)

1. **Code storage in candidates.** Store code only for `status in
   {"validated", "submitted"}`. Sketched-only candidates carry
   `code = None`. Keeps scratchpad lean while preserving the records
   that are actually worth re-reading.

2. **Cap-when-referenced.** No reference-aware logic. Cap at
   `MAX_CANDIDATES = 100` and drop oldest first. Simpler; the larger
   cap absorbs the loss.

3. **Candidate ID format.** Monotonic `cand_0042` via
   `next_candidate_seq`. Easier to reason about in logs and prompts
   than random hashes.

4. **Implicit candidate creation in `validate_code`.** Auto-create a
   candidate ONLY when validation passes. Failed validates do not
   create a record. Sketched candidates passed in via `candidate_id`
   are updated in place and gain `status="validated"` plus the code
   payload.

5. **Submission also writes a history entry.** Today the openai_sdk
   agent never calls `history.add_entry`, so `state["history"]` is
   permanently empty and `merge_results_into_state` has nothing to
   join against. Feature 1 fixes this by having `_submit` add a
   history entry (with `candidate_id`) before raising `SubmitSignal`.
   This is in scope because it's where the `candidate_id` ↔ history
   join lives. Auto-submit recovery (when the LLM never explicitly
   calls submit) does NOT add a history entry — keeping that gap
   matches today's behaviour and avoids creeping scope.

### 5.1 Still open (don't block feature 1)

1. **Cross-round candidate visibility.** Should `list_candidates`
   default to "this round only" or "all-time"? Leaning all-time with a
   `recent_only=True` flag, since the lineage tree's value is
   long-term. Decide before the tool ships in feature 1's PR.

2. **Macro failure semantics.** Step fails → abort vs continue?
   Default abort, but expose `continue_on_error: bool = False` per
   step? I'd start with always-abort to keep the spec small.

3. **Macro nesting.** Can a macro call `run_macro`? Risk: infinite
   recursion. Easy fix: forbid `run_macro` and `define_macro` from
   appearing inside a macro `sequence` (validate at define time).

4. **`previous_results` → hypothesis outcome auto-link.** This needs
   the join key. We have `history.candidate_id` and
   `hypothesis.candidate_id`, so the join is
   `hypothesis.candidate_id == history.candidate_id`, then read the
   score off the history entry after `merge_results_into_state` runs.
   Verdict (supported/refuted/inconclusive) — auto-derive from rank
   relative to frontier? Or leave as `"inconclusive"` until the LLM
   judges manually? **My recommendation:** auto-fill score/rank only;
   leave verdict to the LLM via `link_hypothesis`.

5. **Notes vs hypotheses double-write.** During the migration window,
   `write_scratchpad(hypothesis=...)` writes both to
   `notes.open_hypotheses` (list[str]) and `hypotheses` (structured).
   When do we stop the double-write? Probably after one round of
   in-the-wild verification.

6. **Token budget impact on `read_scratchpad`.** Adding the
    `Hypotheses` and `Candidates` sections inflates the read result.
    `read_scratchpad` already returns up to ~10 history entries; we
    should probably cap each new section at 10 visible items too, with
    "(N more, use list_candidates/...)" footers.

---

## 6. Implementation order (for reference, not part of the schema)

1. **Feature 1** lands the `candidates` list, the seq counter, and the
   modified tool signatures. Adds `read_my_submissions` would come now
   too if cheap — it's just a read against existing state plus the new
   `candidate_id` field on history.
2. **Feature 2** finalizes `read_my_submissions` (the modification of
   history entries to carry `candidate_id` was done in feature 1).
3. **Feature 3** adds `hypotheses` and the auto-link pass.
4. **Feature 4** adds `macros`.

After feature 1 is in, the schema for the rest is a matter of adding
keys, not redesigning.
