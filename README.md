# Radar Miner Agents

Competitive miner agents for the Radar Bittensor subnet. Each agent uses the code-injection model — submit `.py` files that run inside the subnet's official agent image.

## Agents

### 1. Frontier Sniper
Wins rounds by barely beating the frontier — surgical micro-improvements to existing winning code. Exploits the steep sigmoid scoring (steepness=20) where even 1-2% improvement maps to a dominant score.

### 2. Bucket Specialist
Exploits the size bucket rotation system by maintaining pre-optimized architecture templates for each FLOPs range. Dominates specific buckets rather than being mediocre across all of them.

### 3. Pareto Hunter
Exploits the 1.5x Pareto dominance bonus by targeting ALL objectives (crps, mase, exec_time, memory_mb). Attacks secondary metrics where the frontier is weakest.

## Structure

Each agent is a **self-contained directory** that can be passed directly to `--agent_dir`:

```
agents/
  frontier_sniper/       ← pass this dir to --agent_dir
    agent.py             ← entry point: design_architecture(challenge, client)
    core/
      llm.py             # LLM client (via GatedClient)
      db_client.py       # Validator DB queries (via GatedClient)
      validation.py      # AST-based code validation
      prompt_builder.py  # Task-aware prompt construction
      history.py         # Persistent experiment history + scratchpad state
  bucket_specialist/
    agent.py
    core/...
  pareto_hunter/
    agent.py
    core/...
tests/
  ...
```

## Deploying

Use `start_miner.sh` instead of calling `neuron.py` directly — it checks DNS resolution with exponential backoff before launching, preventing crash loops when the network is temporarily unavailable.

```bash
# Agent 1
./start_miner.sh --agent_dir agents/frontier_sniper/ \
    --wallet.name miner1 --netuid <N> --subtensor.network <network>

# Agent 2
./start_miner.sh --agent_dir agents/bucket_specialist/ \
    --wallet.name miner2 --netuid <N> --subtensor.network <network>

# Agent 3
./start_miner.sh --agent_dir agents/pareto_hunter/ \
    --wallet.name miner3 --netuid <N> --subtensor.network <network>
```

### PM2 (recommended)

Use the included `ecosystem.config.js` which adds exponential backoff restart delays:

```bash
# Edit ecosystem.config.js to set your wallet names and netuid, then:
pm2 start ecosystem.config.js

# Or start a single miner:
pm2 start ecosystem.config.js --only miner_1
```

The harness volume-mounts the agent directory to `/workspace/agent/` and calls `design_architecture(challenge, client)` from `agent.py`.

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
- **Available libraries**: Python 3.11 stdlib + numpy only

### Proxy Endpoints (from challenge dict)

| Key | Usage |
|-----|-------|
| `challenge["db_url"]` | `client.get_json(f"{url}/experiments/recent")` |
| `challenge["desearch_url"]` | `client.post_json(f"{url}/search", {"query": "...", "max_results": 5})` |
| `challenge["llm_url"]` | `client.post_json(f"{url}/chat", {"model": "...", "messages": [...], "temperature": 0.7, "max_tokens": 4096})` |
| `challenge["llm_url"]/models` | `client.get_json(...)` to list allowed models |

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
