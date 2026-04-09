# Patch Decoder Agent

> See the root [CLAUDE.md](../../CLAUDE.md) for full challenge spec, code requirements, and environment details.

## Strategy

Deterministic agent — NO LLM calls. Generates a RevIN patch-MLP decoder that is
analytically scaled to fit the requested FLOPs budget. Pure code generation via
grid search over (d_model, n_layers, patch_size).

## Key Design Decisions

- **Zero LLM dependency**: works even when LLM service is down or slow
- **Analytical FLOPs estimation**: `_analytical_flops()` computes exact FLOPs for any config
- **Grid search**: scans patch_size in [64,32,16,8,4], n_layers in [1,2,3,4,6], d_model in [4..512 step 4]
- **Target 60% of max FLOPs**: picks the config closest to this target
- **Self-correcting**: if validation fails, adjusts d_model by +-4/8/12 and retries

## Architecture

1. **RevIN normalization** (learnable affine per-variate)
2. **Non-overlapping patch embedding** (Linear: patch_size -> d_model)
3. **Learnable positional embedding**
4. **N stacked MLP blocks** (Linear -> LayerNorm -> GELU -> Linear + residual)
5. **Flatten patches -> projection head** -> (pred_len, num_variates, n_quantiles)
6. **RevIN denormalization** (applied per-quantile to preserve last-dim = num_variates)

## Entry Point

`agents/patch_decoder/agent.py:design_architecture`

### Flow

1. Identify FLOPs bucket from challenge
2. `_compute_scaling(challenge)` — grid search for optimal (d_model, n_layers, patch_size)
3. `_generate_code(cfg)` — render complete Python module from config
4. Validate code; if invalid, adjust d_model by deltas [-4,4,-8,8,-12,12] and retry
5. Update scratchpad with history entry (strategy="deterministic")

## Core Modules (Subset — No LLM)

| Module | Purpose |
|--------|---------|
| `core/validation.py` | AST-based code validation (`validate_code` variant) |
| `core/history.py` | Scratchpad state, bucket identification, FLOPs budget extraction |

Note: This agent does NOT use `core/llm.py`, `core/db_client.py`, `core/prompt_builder.py`, or `core/tools.py`.

## Generated Code Includes

- `build_model(context_len, prediction_len, num_variates, quantiles)` — required
- `build_optimizer(model)` — AdamW with size-scaled LR
- `training_config()` — batch_size and grad_accum scaled by model size
- `build_scheduler(optimizer, total_steps)` — cosine with warmup
- `init_weights(model)` — Xavier uniform

## What NOT to Change

- Do not add LLM calls — the deterministic nature is the core value proposition
- The `_analytical_flops()` function must stay in sync with the actual generated architecture
- The grid search order (patch_size outer, d_model inner) is optimized for early stopping
- Training hyperparameters (LR, batch_size, grad_accum) are already size-scaled in `_compute_scaling()`
