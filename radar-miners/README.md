# Radar Miner Agents

Competitive miner agents for the Radar Bittensor subnet. Each agent uses [Chutes](https://llm.chutes.ai/v1) for LLM inference and employs orthogonal strategies to maximize mining rewards.

## Agents

### 1. Frontier Sniper
Wins rounds by barely beating the frontier — surgical micro-improvements to existing winning code. Exploits the steep sigmoid scoring (steepness=20) where even 1-2% improvement maps to a dominant score.

### 2. Bucket Specialist
Exploits the size bucket rotation system by maintaining pre-optimized architecture templates for each FLOPs range. Dominates specific buckets rather than being mediocre across all of them.

### 3. Pareto Hunter
Exploits the 1.5x Pareto dominance bonus by targeting ALL objectives (crps, mase, exec_time, memory_mb). Attacks secondary metrics where the frontier is weakest.

## Structure

```
radar-miners/
  core/
    llm.py            # Chutes LLM client (OpenAI-compatible)
    scratchpad.py      # R2 scratchpad load/save
    db_client.py       # Validator DB query client
    validation.py      # AST-based code validation
    prompt_builder.py  # Task-aware prompt construction
    history.py         # Persistent experiment history
  agents/
    frontier_sniper/
      run.py
      Dockerfile
    bucket_specialist/
      run.py
      Dockerfile
    pareto_hunter/
      run.py
      Dockerfile
  tests/
    ...
```

## Environment Variables

- `CHUTES_API_KEY` (required) — API key for Chutes LLM
- `CHUTES_MODEL` (optional) — Model to use (default: `deepseek-ai/DeepSeek-V3-0324`)

## Building & Running

```bash
# Build an agent
cd radar-miners
docker build -f agents/frontier_sniper/Dockerfile -t frontier-sniper .

# Run with challenge JSON on stdin
echo '{"task": {...}, "flops_budget": {...}}' | \
  docker run -i -e CHUTES_API_KEY=your_key frontier-sniper
```

## Testing

```bash
cd radar-miners
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
