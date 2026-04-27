# RADAR Subnet Agent — CLAUDE.md

## What This Is

This repo contains a miner agent for the RADAR Bittensor subnet. Validators
call `design_architecture(challenge, client)` every round. The agent must
return Python code that defines a model architecture.

## Entry Point

```python
def design_architecture(challenge: dict, client: GatedClient) -> dict:
    """Called by the validator sandbox every round.
    
    Returns: {"code": str, "name": str, "motivation": str}
    """
```

`challenge` — round parameters (task spec, FLOPs range, frontier, service URLs)
`client` — GatedClient, the ONLY way to make HTTP requests (no requests/httpx/aiohttp available)
- `client.get_json(url)` → dict
- `client.post_json(url, payload)` → dict
- `client.get(url)` → bytes
- `client.put(url, data, content_type=...)` → None

## The Challenge Dict

```python
challenge = {
    # ── Round identity ──
    "challenge_id": str,
    "round_id": int,
    "seed": int,
    "eval_split_seed": int,

    # ── FLOPs budget (HARD gate — model outside range = rejected) ──
    "min_flops_equivalent": int,
    "max_flops_equivalent": int,

    # ── Task spec (NEVER hardcode — always read from here) ──
    "task": {
        "name": str,                    # e.g. "ts_forecasting"
        "description": str,
        "task_params": dict,            # DEFINES build_model() signature
        "constraints": list[str],       # Rules the code MUST follow
        "objectives": list[dict],       # What gets scored
        "domain_system_prompt": str,    # Domain context for LLM prompting
        "anti_patterns": list[str],     # Things to avoid
        "example_hypotheses": list[str],# Ideas for inspiration
        "time_budget": int,             # Training wall-clock seconds
        "runner_dir": str,              # Which runner executes the code
    },

    # ── Service URLs (use via client) ──
    "db_url": str,         # experiment database (GET /experiments/recent?limit=10)
    "desearch_url": str,   # arxiv search (POST /search {query, count, date_filter})
    "llm_url": str,        # LLM inference (POST /chat, GET /models)
    "agent_token": str,    # auto-injected into client headers

    # ── Pareto frontier (filtered to this round's size bucket) ──
    "feasible_frontier": [
        {"code": str, "metric": float, "objectives": dict,
         "parent_diff": str, "motivation": str},
    ],

    # ── Scratchpad (persistent private storage across rounds) ──
    "scratchpad_get_url": str,
    "scratchpad_put_url": str,
    "scratchpad_max_mb": int,
}
```

## What Code Must Contain

The returned code string is validated by AST parsing. Missing required
functions = instant rejection before training even starts.

### Required (MUST be top-level function defs)

```python
def build_model(**task_params) -> nn.Module:
    # Signature comes from challenge["task"]["task_params"] keys.
    # Example for ts_forecasting: build_model(context_len, prediction_len, num_variates, quantiles)
    # NEVER hardcode — read the keys from the challenge.
    ...

def build_optimizer(model) -> torch.optim.Optimizer:
    ...
```

### Optional hooks (harness checks for these with hasattr)

```python
def init_weights(model) -> None:                    # Custom init (param count must NOT change)
def transform_batch(batch, step, total_steps) -> dict:  # Data augmentation (batch has "input"/"target")
def on_step_end(model, optimizer, step, total_steps, loss_value) -> None:  # EMA, freezing, etc.
def configure_amp() -> dict:                         # {"enabled": bool, "dtype": "bfloat16"|"float16"|"float32"}
def compute_loss(predictions, targets) -> Tensor:    # Custom loss
def build_scheduler(optimizer, total_steps):         # LR scheduler
def training_config() -> dict:                       # {"batch_size": int, "grad_accum_steps": int, "grad_clip": float}
COMPILE = True                                       # Module-level bool for torch.compile
```

## How to Read task_params

The keys of `challenge["task"]["task_params"]` ARE the positional args to `build_model()`:

```python
task_params = challenge["task"]["task_params"]
# task_params.keys() → the build_model signature
# e.g. {"context_len": 512, "prediction_len": 96, "num_variates": 1, "quantiles": [...]}
#   → def build_model(context_len, prediction_len, num_variates, quantiles)
```

Current ts_forecasting task_params:
- `context_len` (int): input sequence length
- `prediction_len` (int): forecast horizon
- `num_variates` (int): number of input channels
- `quantiles` (list[float]): quantile levels for probabilistic output

I/O shapes for ts_forecasting:
- Input: `(batch, context_len, num_variates)`
- Output: `(batch, prediction_len, num_variates, len(quantiles))`

Other tasks will have different task_params. Always read from the challenge.

## FLOPs Budget

Models are measured with `torch.utils.flop_counter.FlopCounterMode`.
Must land in `[min_flops_equivalent, max_flops_equivalent]`. Target ~60% of max.

| Bucket       | Min  | Max  |
|-------------|------|------|
| Tiny        | 100K | 500K |
| Small       | 500K | 2M   |
| Medium-small| 2M   | 10M  |
| Medium      | 10M  | 50M  |
| Large       | 50M  | 125M |

## Scoring

1. Size gate (hard reject if FLOPs outside range)
2. Primary objective from `challenge["task"]["objectives"]` (the one with `"primary": true`)
3. Pareto dominance bonus: 1.5x if model dominates existing frontier
4. Penalties: FLOPs mismatch (0.3x), trainer failure/timeout (0.5x)

## Available Services

### LLM (`challenge["llm_url"]`)

```python
# List models
models = client.get_json(f"{llm_url}/models")["models"]
# Chat completion
resp = client.post_json(f"{llm_url}/chat", {
    "model": models[0],
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.7, "max_tokens": 4096,
})
content = resp["content"]
```

### Experiment DB (`challenge["db_url"]`)

```python
experiments = client.get_json(f"{db_url}/experiments/recent?limit=10")
```

### Arxiv Search (`challenge["desearch_url"]`)

```python
results = client.post_json(f"{desearch_url}/search", {"query": "...", "count": 10, "date_filter": "PAST_2_YEARS"})["results"]
```

### Scratchpad (persistent across rounds)

```python
scratch_dir = load_scratchpad(challenge)   # injected by harness
# ... read/write files in scratch_dir ...
save_scratchpad(challenge, scratch_dir)    # injected by harness
```

## Environment

- Runs inside a sandboxed Docker container (official image, not yours)
- Only torch + standard library available (no pip install)
- Network restricted to allowed URLs only (via GatedClient)
- stderr is captured as reasoning trace (use `print(..., file=sys.stderr)`)
- stdout is the JSON result (handled by harness, don't print to stdout)

## Code Rules

- Only torch + stdlib — no external dependencies
- Code must be syntactically valid Python (AST-parsed before execution)
- `build_model` and `build_optimizer` must be top-level def statements
- No hardcoded task params, FLOPs presets, or model signatures
- Always read everything from the challenge dict
