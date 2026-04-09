# Reliable Baseline Agent

> See the root [CLAUDE.md](../../CLAUDE.md) for full challenge spec, code requirements, and environment details.

## Strategy

Safe fallback with robust error handling and graceful degradation. Skips
submission rather than submitting broken code. Every external call is wrapped
in try/except — if the LLM is down, it skips rather than crashes.

## Key Design Decisions

- **Graceful degradation**: LLM failure → skip submission (empty code + "skipped" message)
- **No validation retry loop**: if LLM returns invalid code, skips rather than retrying aggressively
- **Higher-level LLM abstraction**: uses `llm.reason_and_generate()` instead of raw chat
- **Conservative FLOPs target**: 55% of max (vs 60% used by other agents) for extra safety margin
- **All external calls are try/except guarded**: scratchpad, tool analysis, LLM — none are fatal

## Entry Point

`agents/reliable_baseline/agent.py:design_architecture`

### Flow

1. Identify FLOPs bucket — checks both `min_flops_equivalent` and nested `flops_budget` format
2. Load scratchpad state (try/except, non-fatal on failure)
3. Gather context: frontier, DB queries (recent, failures, component stats, dead ends)
4. Optional tool-assisted analysis (try/except, non-fatal)
5. Call `llm.reason_and_generate(client, challenge, context)` — single high-level call
6. If LLM unavailable or fails → skip submission (empty code)
7. Final validation gate — empty code on failure
8. Update scratchpad (try/except on save)

### Differences from Other Agents

- Uses `llm.reason_and_generate()` (high-level) vs raw `llm.chat()` + `llm.extract_code()`
- Uses `validation.validate_code()` variant (not `validation.validate()`)
- No 3-attempt retry loop — delegates retry logic to `reason_and_generate()`
- Targets 55% of max FLOPs (other agents use 60%)

## Core Modules

| Module | Purpose |
|--------|---------|
| `core/llm.py` | `reason_and_generate()` — high-level LLM orchestration |
| `core/db_client.py` | Experiment DB queries |
| `core/validation.py` | AST-based code validation (`validate_code` variant) |
| `core/history.py` | Scratchpad state, bucket identification |
| `core/tools.py` | Tool definitions + handlers for DB research |

Note: Does NOT use `core/prompt_builder.py` — prompt construction is handled inside `reason_and_generate()`.

## Scratchpad State

- Standard history entries (strategy="reliable_baseline")
- No agent-specific state (no playbooks, templates, or dominatable tracking)

## What NOT to Change

- Do not add aggressive retry loops — the "skip rather than submit bad code" philosophy is intentional
- Do not remove try/except guards on external calls — reliability is the core value
- The 55% FLOPs target is intentionally conservative; don't raise it
- Do not replace `reason_and_generate()` with raw chat calls — the abstraction handles retry internally
