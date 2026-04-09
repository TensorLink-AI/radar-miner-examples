# Bucket Specialist Agent

> See the root [CLAUDE.md](../../CLAUDE.md) for full challenge spec, code requirements, and environment details.

## Strategy

Dominate specific FLOPs ranges with pre-optimized architecture templates.
Maintains a per-bucket template library that evolves over rounds. Once on the
frontier, pivots to secondary objectives (exec_time, memory) for the 1.5x
Pareto bonus.

## Key Design Decisions

- **Per-bucket template library**: saved winning code is reused as starting point in future rounds
- **Dynamic sizing guidance**: computes max hidden dim from challenge params per bucket
- **EMA alpha = 0.3**: winning 2-3 out of 5 rounds sustains strong weight over time
- **Target exactly 60% of max FLOPs** for safety margin within the 10% tolerance gate
- **Secondary objective pivot**: if already on frontier, optimize exec_time/memory for 1.5x bonus

## Entry Point

`agents/bucket_specialist/agent.py:design_architecture`

### Flow

1. Identify FLOPs bucket and load scratchpad state
2. Get frontier + query experiment DB
3. Compute dynamic bucket guidance (max hidden dim, FLOPs budget tips)
4. Retrieve saved template for this bucket from scratchpad (if any)
5. Optional tool-assisted research (LLM with DB tools, bucket-focused)
6. LLM generation with 3-attempt validation loop
7. Final validation gate — rejects invalid code
8. Updates scratchpad: history entry + saves code as bucket template via `save_template()`

## Core Modules

| Module | Purpose |
|--------|---------|
| `core/llm.py` | LLM chat + code extraction via GatedClient |
| `core/db_client.py` | Experiment DB queries |
| `core/validation.py` | AST-based code validation |
| `core/prompt_builder.py` | System/user prompt construction |
| `core/history.py` | Scratchpad state, bucket history, FLOPs budget extraction |
| `core/flops_estimator.py` | Analytical FLOPs estimation |
| `core/tools.py` | Tool definitions + handlers for DB research |

## Scratchpad State

- `templates` dict: per-bucket saved code (best architecture for each bucket)
- `template_metrics` dict: per-bucket metrics (crps, flops) from previous wins
- Standard history entries

## What NOT to Change

- Do not remove per-bucket template persistence — that's the core competitive advantage
- Do not merge bucket strategies — each bucket needs independent optimization
- The `_compute_bucket_guidance()` function dynamically sizes from challenge params; don't hardcode
- The `save_template()` call after submission is critical for cross-round learning
