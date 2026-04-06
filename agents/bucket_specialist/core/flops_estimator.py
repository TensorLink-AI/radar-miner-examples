"""Estimate FLOPs for generated model code by executing build_model and counting ops."""

import ast
import sys
import torch
import torch.nn as nn

# Default harness parameters (match the validator's documented interface)
DEFAULT_CONTEXT_LEN = 512
DEFAULT_PREDICTION_LEN = 96
DEFAULT_NUM_VARIATES = 370
DEFAULT_QUANTILES = [0.1, 0.5, 0.9]


def _count_module_flops(module, input, output):
    """Forward hook that estimates FLOPs for common layer types."""
    flops = 0
    if isinstance(module, nn.Linear):
        # 2 * in * out per element (multiply + accumulate)
        inp = input[0]
        batch_elements = inp.numel() // module.in_features
        flops = 2 * module.in_features * module.out_features * batch_elements
    elif isinstance(module, nn.Conv1d):
        out_len = output.shape[-1]
        flops = (2 * module.in_channels * module.kernel_size[0]
                 * module.out_channels * out_len * output.shape[0]
                 // module.groups)
    elif isinstance(module, nn.Conv2d):
        out_h, out_w = output.shape[2], output.shape[3]
        kh, kw = module.kernel_size
        flops = (2 * module.in_channels * kh * kw
                 * module.out_channels * out_h * out_w * output.shape[0]
                 // module.groups)
    elif isinstance(module, nn.MultiheadAttention):
        # Q, K, V projections + attention scores + output projection
        # Approximate: 4 * seq_len * embed_dim^2 + 2 * seq_len^2 * embed_dim
        inp = input[0]  # (seq_len, batch, embed_dim) or (batch, seq_len, embed_dim)
        if inp.dim() == 3:
            seq_len = inp.shape[0] if not module.batch_first else inp.shape[1]
            batch = inp.shape[1] if not module.batch_first else inp.shape[0]
            embed_dim = module.embed_dim
            flops = batch * (4 * seq_len * embed_dim * embed_dim
                             + 2 * seq_len * seq_len * embed_dim)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        flops = 2 * input[0].numel()
    elif isinstance(module, (nn.GELU, nn.ReLU, nn.SiLU, nn.Sigmoid, nn.Tanh)):
        flops = input[0].numel()

    if not hasattr(module, '_estimated_flops'):
        module._estimated_flops = 0
    module._estimated_flops += flops


def estimate_flops(code: str, challenge: dict) -> tuple[int | None, str]:
    """Execute generated code, build the model, and estimate FLOPs.

    Returns (estimated_flops, error_message).
    On success error_message is empty. On failure estimated_flops is None.
    """
    # Extract harness params from challenge, falling back to defaults
    task = challenge.get("task", {})
    context_len = task.get("context_len", DEFAULT_CONTEXT_LEN)
    prediction_len = task.get("prediction_len", DEFAULT_PREDICTION_LEN)
    num_variates = task.get("num_variates", DEFAULT_NUM_VARIATES)
    quantiles = task.get("quantiles", DEFAULT_QUANTILES)

    # Execute the code in a restricted namespace
    namespace = {}
    try:
        exec(compile(code, "<generated>", "exec"), namespace)
    except Exception as exc:
        return None, f"Code execution failed: {exc}"

    build_model_fn = namespace.get("build_model")
    if not callable(build_model_fn):
        return None, "build_model not found or not callable"

    # Instantiate the model
    try:
        model = build_model_fn(context_len, prediction_len, num_variates, quantiles)
    except Exception as exc:
        return None, f"build_model() raised: {exc}"

    if not isinstance(model, nn.Module):
        return None, "build_model() did not return an nn.Module"

    model.eval()

    # Register hooks on leaf modules
    hooks = []
    for m in model.modules():
        hooks.append(m.register_forward_hook(_count_module_flops))

    # Forward pass with dummy input: (batch=1, context_len, num_variates)
    try:
        with torch.no_grad():
            dummy = torch.randn(1, context_len, num_variates)
            model(dummy)
    except Exception as exc:
        for h in hooks:
            h.remove()
        return None, f"Forward pass failed: {exc}"

    # Sum FLOPs across all modules
    total_flops = 0
    for m in model.modules():
        total_flops += getattr(m, '_estimated_flops', 0)

    # Clean up
    for h in hooks:
        h.remove()
    for m in model.modules():
        if hasattr(m, '_estimated_flops'):
            del m._estimated_flops

    return total_flops, ""
