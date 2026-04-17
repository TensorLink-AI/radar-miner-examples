# OpenAI SDK reference agent

A reference implementation of a Radar miner that uses the OpenAI Python
SDK (`openai>=1.50`, installed in the official agent image) instead of
the hand-rolled urllib that the `autonomous/` agent uses. Network egress
is enforced at the network layer (iptables) inside the agent pod, so
bypassing `GatedClient` is safe.

## When to use this

Pick this agent as your starting point if you want:

- Automatic retry on transient errors (429, 5xx, timeout) classified by
  the SDK rather than substring-matched on raw exception text
- Clean tool-calling via `ChatCompletionMessage.tool_calls` rather than
  parsing dicts
- Easy swap to another OpenAI-compatible endpoint or model
- Less code to maintain than `autonomous/`

## When to use `autonomous/` instead

- You're on an older agent image without the SDK
- You want full control over the HTTP transport
- You're implementing a novel retry / routing strategy
- You want the strategy framework (personas / per-strategy temperatures)

## Architecture

Three phases:

1. **Research** — first 20% of the budget (capped at 120s). The LLM
   calls `analyze_task`, `list_frontier`, `get_frontier_member`.
2. **Design** — middle 60% of the budget. The LLM iterates on
   `validate_code` until something passes, then calls `submit`.
3. **Reserve** — last 30s. Guaranteed window to wrap up; if nothing
   submitted, ship a guaranteed-valid template via
   `core.fallback_templates.generate_fallback`.

## File layout

```
agents/openai_sdk/
├── __init__.py        # puts this dir on sys.path for package-mode imports
├── agent.py           # entry point — design_architecture
├── llm_client.py      # cached OpenAI client + retry/failover
├── prompts.py         # system/user prompt builders
├── tools.py           # tool schemas + handlers (OpenAI function-call format)
├── validation.py      # re-export of core.validation.validate_code
├── core/              # FLOPs / validation / history / fallback helpers
└── README.md          # this file
```

`core/` lives inside this package (copied from
`agents/autonomous/core/`) so the harness can ship the directory
flat to `/workspace/agent/` and `from core.X import Y` resolves
without any package-import plumbing — exactly like the autonomous
agent is deployed.

## Reused from `agents/autonomous/core/`

- `validation.validate_code` — AST + FLOPs + output-shape checks
- `flops_estimator.estimate_flops` — FlopCounterMode wrapper
- `history.extract_flops_budget`, `history.identify_bucket` — bucket math
- `fallback_templates.generate_fallback` — guaranteed-valid model template
- `prompt_builder._compute_sizing_guidance` — sizing prompt fragments

## Environment variables

| Name | Default | Meaning |
|------|---------|---------|
| `LLM_URL` | (required) | OpenAI-compatible base URL (the proxy) |
| `AGENT_TOKEN` | `""` | Forwarded as `X-Agent-Token` header |
| `MINER_UID` | `"0"` | Forwarded as `X-Miner-UID` header |
| `LLM_READ_TIMEOUT` | `180` | Per-request read timeout in seconds |
| `AGENT_BUDGET_SECONDS` | (none) | Override agent-side time budget |

## Selecting this agent

Miners pick their entry point via `AGENT_MODULE`:

```bash
AGENT_MODULE=/workspace/agent/agents/openai_sdk/agent.py
```

The default remains `agents/autonomous/agent.py` until this version is
clearly at parity in production.

## Running tests locally

```bash
python -m pytest tests/test_openai_sdk_agent.py -v
```

The tests mock the OpenAI client at module level — no network calls are
made.
