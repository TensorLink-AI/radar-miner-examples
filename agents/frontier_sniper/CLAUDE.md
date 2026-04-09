# Frontier Sniper Agent

> See the root [CLAUDE.md](../../CLAUDE.md) for full challenge spec, code requirements, and environment details.

## Strategy

Surgical micro-improvements to barely beat the frontier. Copy the best frontier
member's code almost entirely and make ONE targeted change (LR schedule,
normalization, weight init, optimizer hyperparams). Minimal diff, maximum impact.

## Key Design Decisions

- **Sigmoid steepness = 20**: even 1-2% CRPS improvement gives ~0.55-0.65 score
- **Softmax temperature = 0.1**: a tiny score lead dominates the round
- **Never redesign the architecture** — only tune training dynamics
- **Keep FLOPs within budget** — do NOT add layers or increase hidden dims

## Entry Point

`agents/frontier_sniper/agent.py:design_architecture`

### Flow

1. Identify FLOPs bucket and load scratchpad state
2. Get frontier members and query experiment DB (recent, failures, component stats, dead ends)
3. Build strategy instructions with frontier analysis + per-bucket playbook from scratchpad
4. Optional tool-assisted research phase (LLM with DB tools, max 4 rounds)
5. LLM generation with 3-attempt validation loop (feeds back errors on retry)
6. Final validation gate — rejects invalid code rather than submitting
7. Updates scratchpad: history entry + per-bucket playbook

### Bootstrap (No Frontier)

When no frontier exists, submits a proven baseline targeting 60% of max FLOPs
with standard best practices (LayerNorm, residual connections, cosine LR).

## Core Modules

| Module | Purpose |
|--------|---------|
| `core/llm.py` | LLM chat + code extraction via GatedClient |
| `core/db_client.py` | Experiment DB queries (recent, failures, component stats, dead ends) |
| `core/validation.py` | AST-based code validation against challenge requirements |
| `core/prompt_builder.py` | System/user prompt construction with task context |
| `core/history.py` | Scratchpad state: load/save, bucket history, FLOPs budget extraction |
| `core/flops_estimator.py` | Analytical FLOPs estimation |
| `core/tools.py` | Tool definitions + handlers for agentic DB research |

## Scratchpad State

- `playbooks` dict: per-bucket strategies learned from previous rounds (last 10 entries per bucket)
- Standard history entries (name, code, motivation, bucket, flops, strategy)

## What NOT to Change

- Do not add multi-objective optimization — that's Pareto Hunter's job
- Do not add bucket templates — that's Bucket Specialist's job
- Do not remove the "copy best frontier code" approach — that IS the strategy
- The 3-attempt validation loop with error feedback is critical for reliability
