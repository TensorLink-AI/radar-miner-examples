# Pareto Hunter Agent

> See the root [CLAUDE.md](../../CLAUDE.md) for full challenge spec, code requirements, and environment details.

## Strategy

Exploit the 1.5x Pareto dominance bonus by beating ALL objectives simultaneously.
Most miners only optimize CRPS — this agent attacks the secondary metrics
(exec_time, memory_mb, mase) as a competitive surface.

## Key Design Decisions

- **Multi-objective scoring**: crps (1.0), mase (0.5), exec_time (0.2), memory_mb (0.1)
- **1.5x dominance bonus** is the primary advantage — must beat ALL metrics of a frontier member
- **Efficiency hooks are mandatory**: `configure_amp()`, `training_config()`, `init_weights()`
- **Weakness analysis**: identifies "dominatable" frontier members weak on secondary metrics
- **Joint loss optimization**: uses `compute_loss()` to optimize crps AND mase together

## Entry Point

`agents/pareto_hunter/agent.py:design_architecture`

### Flow

1. Identify FLOPs bucket and load scratchpad state
2. Get frontier + query experiment DB
3. Analyze frontier weaknesses: score each member on secondary metric gaps
4. Build strategy instructions with weakness analysis + dominatable target count
5. Append efficiency requirements addendum (configure_amp, training_config, init_weights, compute_loss)
6. Optional tool-assisted research (LLM with DB tools, efficiency-focused)
7. LLM generation with 3-attempt validation loop (retry reminds about efficiency hooks)
8. Final validation gate — rejects invalid code
9. Updates scratchpad with history entry

### Efficiency Requirements (always appended to prompt)

Generated code MUST include:
1. `configure_amp()` returning `{'enabled': True, 'dtype': 'bfloat16'}` — memory savings
2. `training_config()` with larger batch_size — faster wall-clock
3. `init_weights(model)` with proper init — fast convergence
4. `compute_loss()` to jointly optimize crps AND mase

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

- `dominatable` dict: per-bucket count of dominatable frontier members
- Standard history entries (filtered to `strategy="pareto_hunter"` for display)

## What NOT to Change

- Do not remove the efficiency addendum from the user prompt — it's essential for multi-objective
- Do not simplify to single-objective CRPS optimization — that's Simple Modeler's approach
- The `analyze_frontier_weaknesses()` function is the core intelligence; preserve its weakness scoring
- The retry prompt specifically reminds about configure_amp/training_config/init_weights
