"""Estimate FLOPs for generated model code via static module-tree analysis.

Instead of running a forward pass (which can OOM on large models or execute
arbitrary code paths), we instantiate the model and walk its nn.Module tree,
computing FLOPs from weight shapes and the known input dimensions.

For layers whose output shape depends on the input (Conv, Attention), we
propagate shapes symbolically through the tree.  This is approximate but safe
— no tensors are allocated beyond the model parameters themselves.
"""

import torch
import torch.nn as nn

# Default harness parameters (match the validator's documented interface)
DEFAULT_CONTEXT_LEN = 512
DEFAULT_PREDICTION_LEN = 96
DEFAULT_NUM_VARIATES = 370
DEFAULT_QUANTILES = [0.1, 0.5, 0.9]

# Maximum parameter memory (bytes) we'll allow build_model to allocate.
# 512 MB covers even the largest bucket (125M FLOPs) with headroom.
_MAX_PARAM_BYTES = 512 * 1024 * 1024


def _static_linear_flops(module: nn.Linear, seq_len: int) -> int:
    """FLOPs for a Linear layer: 2 * in * out per sequence element."""
    return 2 * module.in_features * module.out_features * seq_len


def _static_conv1d_flops(module: nn.Conv1d, seq_len: int) -> int:
    """FLOPs for Conv1d: 2 * C_in * K * C_out * L_out / groups."""
    padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
    dilation = module.dilation[0] if isinstance(module.dilation, tuple) else module.dilation
    stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
    k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
    out_len = (seq_len + 2 * padding - dilation * (k - 1) - 1) // stride + 1
    return 2 * module.in_channels * k * module.out_channels * max(out_len, 1) // module.groups


def _static_conv2d_flops(module: nn.Conv2d, h: int, w: int) -> int:
    """FLOPs for Conv2d with spatial dims h x w."""
    ph, pw = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
    dh, dw = module.dilation if isinstance(module.dilation, tuple) else (module.dilation, module.dilation)
    sh, sw = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
    kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
    oh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    return 2 * module.in_channels * kh * kw * module.out_channels * max(oh, 1) * max(ow, 1) // module.groups


def _static_mha_flops(module: nn.MultiheadAttention, seq_len: int) -> int:
    """FLOPs for MultiheadAttention: QKV projections + attention + output proj."""
    d = module.embed_dim
    # 3 projections (Q, K, V) + output projection: 4 * 2 * seq * d * d
    proj_flops = 4 * 2 * seq_len * d * d
    # Attention scores: 2 * seq^2 * d  (QK^T + softmax·V)
    attn_flops = 2 * seq_len * seq_len * d
    return proj_flops + attn_flops


def _count_param_bytes(model: nn.Module) -> int:
    """Total bytes used by model parameters."""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return total


def _walk_flops(model: nn.Module, seq_len: int) -> int:
    """Walk the module tree and sum FLOPs from leaf layers.

    Uses seq_len as the token/sequence dimension (num_variates for our harness,
    since the input is transposed to (batch, num_variates, context_len) before
    hitting Linear layers, or kept as (batch, context_len, num_variates)).

    We count each leaf module once. For simplicity we assume the sequence
    dimension is preserved (true for most time-series architectures that avoid
    pooling). This is approximate but sufficient for bucket gate-keeping.
    """
    total = 0
    for m in model.modules():
        # Only count leaf-ish modules (the ones that do actual compute)
        if isinstance(m, nn.Linear):
            total += _static_linear_flops(m, seq_len)
        elif isinstance(m, nn.Conv1d):
            total += _static_conv1d_flops(m, seq_len)
        elif isinstance(m, nn.Conv2d):
            # Assume spatial dims are roughly sqrt(seq_len) x sqrt(seq_len)
            side = max(int(seq_len ** 0.5), 1)
            total += _static_conv2d_flops(m, side, side)
        elif isinstance(m, nn.MultiheadAttention):
            total += _static_mha_flops(m, seq_len)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                            nn.GroupNorm, nn.InstanceNorm1d)):
            # ~2 ops per element (normalize + scale/shift)
            norm_features = getattr(m, 'normalized_shape', None)
            if isinstance(norm_features, (list, tuple)):
                elem = 1
                for s in norm_features:
                    elem *= s
            else:
                elem = getattr(m, 'num_features', seq_len)
            total += 2 * elem * seq_len
    return total


def estimate_flops(code: str, challenge: dict) -> tuple[int | None, str]:
    """Execute build_model, then statically estimate FLOPs from the module tree.

    Returns (estimated_flops, error_message).
    On success error_message is empty. On failure estimated_flops is None.
    """
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

    # Instantiate the model (weights only — no input tensors)
    try:
        model = build_model_fn(context_len, prediction_len, num_variates, quantiles)
    except Exception as exc:
        return None, f"build_model() raised: {exc}"

    if not isinstance(model, nn.Module):
        return None, "build_model() did not return an nn.Module"

    # Safety check: reject models that are absurdly large before walking
    param_bytes = _count_param_bytes(model)
    if param_bytes > _MAX_PARAM_BYTES:
        del model
        return None, (
            f"Model parameters use {param_bytes / (1024**2):.0f} MB, "
            f"exceeding the {_MAX_PARAM_BYTES // (1024**2)} MB safety limit"
        )

    # Static FLOPs estimate — walk module tree, no forward pass needed.
    # Use num_variates as the sequence dimension (harness transposes input
    # to (batch, num_variates, context_len) before the first Linear).
    seq_len = num_variates
    total_flops = _walk_flops(model, seq_len)

    del model
    return total_flops, ""
