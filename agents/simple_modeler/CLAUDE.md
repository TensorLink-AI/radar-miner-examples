# Simple Modeler Agent

> See the root [CLAUDE.md](../../CLAUDE.md) for full challenge spec, code requirements, and environment details.

## Strategy

Pragmatic ML engineering — build a good model, no gaming or overthinking.
No bucket specialization, no multi-objective gaming, no frontier sniping.
Just a solid, well-designed PyTorch model that fits within the FLOPs budget.

## Key Design Decisions

- **No strategy preamble gaming**: tells the LLM to "just build a solid model"
- **Frontier shown for inspiration only**: displays best CRPS but doesn't instruct to copy code
- **Dynamic sizing from challenge params**: computes max hidden dim but doesn't over-constrain
- **Standard 3-attempt validation loop**: straightforward retry with error feedback
- **No per-bucket specialization**: same approach regardless of bucket

## Entry Point

`agents/simple_modeler/agent.py:design_architecture`

### Flow

1. Identify FLOPs bucket and load scratchpad state
2. Get frontier + query experiment DB
3. Build simple strategy instructions (target FLOPs, sizing guidance, best frontier CRPS)
4. Optional tool-assisted research (LLM with DB tools)
5. LLM generation with 3-attempt validation loop
6. Final validation gate — rejects invalid code
7. Updates scratchpad with history entry (strategy="simple_modeler")

### Prompt Philosophy

- Shows frontier's best CRPS for context but says "don't just copy it"
- Shows bucket history to avoid repeating failures
- No playbooks, no templates, no weakness analysis
- Tells the LLM: "a working simple model beats a broken clever one"

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

- Standard history entries (strategy="simple_modeler")
- No agent-specific state (no playbooks, templates, or special tracking)

## What NOT to Change

- Do not add complex strategy logic — simplicity IS the strategy
- Do not add per-bucket template persistence — that's Bucket Specialist's job
- Do not add frontier weakness analysis — that's Pareto Hunter's job
- Do not instruct the LLM to "copy and modify" frontier code — that's Frontier Sniper's job
- The straightforward prompt style is intentional; don't over-engineer it
