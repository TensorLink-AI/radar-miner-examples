# Radar Miner Agents

Competitive miner agents for the Radar Bittensor subnet. Each agent uses the
code-injection model — submit `.py` files that run inside the subnet's official
agent image.

## Agents

### `autonomous/` — Tool-using LLM with strategy framework

The default agent. A single autonomous loop where the LLM picks tools (research,
sketch, validate, submit) until it ships a model. A **strategy** is a personality
prompt — the validation loop, tools, and fallback chain are shared. One strategy
is selected per round based on prior history and the current frontier.

Strategies (selected via `strategies.select_strategy(challenge, state)`):

| Strategy             | When it's picked (default) | What it pushes the LLM to do |
|----------------------|----------------------------|------------------------------|
| `reliable_baseline`  | No frontier exists yet     | Conservative bootstrap       |
| `simple_modeler`     | Default                    | Build a clean working model  |
| `frontier_sniper`    | Have prior code in bucket  | Surgical micro-improvements  |
| `ensemble_distiller` | Frontier ≥ 3 members       | Cherry-pick best operational choice per role |
| `ablation_scientist` | ≥5 entries in bucket       | One controlled experiment per round |
| `training_optimizer` | Plateau in bucket          | Freeze architecture, change training dynamics |
| `bucket_specialist`  | Override only               | Evolve the bucket's saved best |
| `pareto_hunter`      | Override only               | Attack the weakest scored objective |

`bucket_specialist` and `pareto_hunter` are opt-in via
`state['strategy_override']` written into the scratchpad.

The selector is task-agnostic — it reads only the FLOPs bucket and the
challenge's frontier/history. New tasks (vision, NLP, graph) work without
strategy changes because each strategy reads task params and constraints from
the challenge dict directly.

### `patch_decoder/` — Frontier-evolution agent

Generates a unified diff against the best frontier code instead of a fresh
model. Specialised sibling to `autonomous` for rounds where small surgical
patches dominate.

## Structure

Each agent is a self-contained directory that can be passed directly to
`--agent_dir`:

```
agents/
  autonomous/                         ← default; pass this dir to --agent_dir
    agent.py                          ← entry point: design_architecture(challenge, client)
    tools.py                          ← tool definitions + handlers + build_tools(challenge)
    core/
      arch_knowledge.py               # operation reference + frontier-gap analysis
      fallback_templates.py           # task-agnostic guaranteed-valid templates
      flops_estimator.py              # torch FlopCounterMode wrapper
      history.py                      # scratchpad state, buckets, history entries
      input_shape.py                  # task-agnostic input shape inference
      output_shape.py                 # task-agnostic output shape inference
      prompt_builder.py               # task params / sizing / frontier formatting
      trace.py                        # per-layer shape + FLOPs trace
      validation.py                   # AST + FLOPs validation
    strategies/
      __init__.py                     # registry + select_strategy + build_strategy
      reliable_baseline.py
      simple_modeler.py
      frontier_sniper.py
      ensemble_distiller.py
      ablation_scientist.py
      training_optimizer.py
      bucket_specialist.py
      pareto_hunter.py
  patch_decoder/
    agent.py
    core/...
tests/
  ...
```

## Deploying

```bash
python miner/neuron.py --agent_dir agents/autonomous/ \
    --wallet.name miner1 --netuid <N> --subtensor.network <network>

# Frontier-patch sibling
python miner/neuron.py --agent_dir agents/patch_decoder/ \
    --wallet.name miner2 --netuid <N> --subtensor.network <network>
```

The harness volume-mounts the agent directory to `/workspace/agent/` and calls
`design_architecture(challenge, client)` from `agent.py`.

## Code-Injection Interface

Each agent must define in `agent.py`:

```python
def design_architecture(challenge: dict, client: GatedClient) -> dict:
    """Returns {"code": str, "name": str, "motivation": str}"""
```

- **`client`** — `GatedClient` is the only way to make HTTP requests:
  - `client.get(url) → bytes`
  - `client.get_json(url) → dict`
  - `client.post(url, data) → bytes`
  - `client.post_json(url, payload) → dict`
  - `client.put(url, data) → int`
- **`load_scratchpad(challenge)`** / **`save_scratchpad(challenge, scratch_dir)`** — injected into module namespace by the harness (don't define them, just call them)
- **No** `main()`, `if __name__`, stdin/stdout, Dockerfile, or requirements.txt
- **Available libraries**: Python 3.11 stdlib + numpy + torch

### Proxy Endpoints (from challenge dict)

| Key | Usage |
|-----|-------|
| `challenge["db_url"]` | `client.get_json(f"{url}/experiments/recent")` |
| `challenge["desearch_url"]` | `client.post_json(f"{url}/search", {"query": "...", "count": 10, "date_filter": "PAST_2_YEARS"})` |
| `challenge["llm_url"]` | `client.post_json(f"{url}/v1/chat/completions", {"model": "...", "messages": [...], "temperature": 0.7, "max_tokens": 4096})` |
| `challenge["llm_url"]/v1/models` | `client.get_json(...)` to list allowed models |

## How the Agent Response Reaches the Validator

Understanding the end-to-end flow prevents the most common rejection: a missing
`build_model()` definition.

### 1. Validator launches the agent pod

The validator calls `launch_agent_pod()` (in `validator/collection.py`) to spin
up a sandboxed container with the miner's agent directory mounted at
`/workspace/agent/`. It then calls `run_agent_on_pod()`, passing the challenge
as JSON.

### 2. Harness runs inside the pod

The official harness (`runner/agent/harness.py`) inside the container:

1. Imports the miner's `agent.py` module.
2. Calls `agent_mod.design_architecture(challenge, client)`.
3. The agent function returns a dict: `{"code": str, "name": str, "motivation": str}`.
4. The harness prints the result as JSON to **stdout**.

### 3. Validator captures stdout

`run_agent_on_pod()` reads the JSON from stdout. If the result contains a
`"code"` key it wraps it in a `Proposal`, uploads the code to R2, and returns
the proposal.

### 4. Pre-validation checks

Before any training or scoring, the validator runs
`pre_validate_code(proposal.code)`. This step AST-parses the `code` string and
checks for top-level `def build_model(...)` and `def build_optimizer(model)`.
If either is missing the proposal is rejected immediately.

`build_model`'s positional argument names come from
`challenge["task"]["task_params"].keys()` and change between tasks — never
hardcode them.

### How `autonomous/` guards against this

The agent's tool loop has a strict contract:

1. The LLM iteratively designs with `analyze_task` → `estimate_layer_flops` /
   `sketch_architecture` → `validate_code` → `submit`.
2. `validate_code` runs the same AST + FLOPs + output-shape checks the
   validator does.
3. If the loop exhausts time or turns without a fully-validated submission,
   the agent falls through to a guaranteed-valid template
   (`core/fallback_templates.py`) sized for the current bucket.

## Testing

```bash
python -m pytest tests/ -v
```

## Key Numbers

| Parameter | Value |
|-----------|-------|
| Size buckets | 5 (tiny, small, medium-small, medium, large) |
| Sigmoid steepness | 20 (5% improvement → ~0.73 score) |
| Softmax temperature | 0.1 (winner takes almost all) |
| EMA alpha | 0.3 (30% new, 70% history) |
| Pareto dominance bonus | 1.5x multiplier |
| FLOPs target | 60% of bucket max |
| FLOPs tolerance | ±10% of bucket bounds |
