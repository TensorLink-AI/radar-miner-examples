"""AST-based code validation — syntax, required functions, forbidden imports, FLOPs bounds."""

import ast
import sys

from core.flops_estimator import estimate_flops
from core.history import extract_flops_budget

FORBIDDEN_IMPORTS = {"subprocess", "socket", "ftplib"}

HARNESS_REQUIRED = {
    "build_model": ["context_len", "prediction_len", "num_variates", "quantiles"],
    "build_optimizer": ["model"],
}

HARNESS_OPTIONAL = [
    "training_config", "init_weights", "configure_amp",
    "transform_batch", "on_step_end", "build_scheduler", "compute_loss",
]


FALLBACK_CODE = '''\
import torch
import torch.nn as nn


class _FallbackModel(nn.Module):
    """Minimal linear model used when the LLM fails to produce valid code."""

    def __init__(self, context_len, prediction_len, num_variates, n_quantiles):
        super().__init__()
        self.prediction_len = prediction_len
        self.n_quantiles = n_quantiles
        self.linear = nn.Linear(context_len, prediction_len * n_quantiles)

    def forward(self, x):
        # x: (batch, context_len, num_variates)
        out = self.linear(x.transpose(1, 2))  # (batch, num_variates, pred*q)
        b, v, _ = out.shape
        out = out.view(b, v, self.prediction_len, self.n_quantiles)
        return out.permute(0, 2, 1, 3)  # (batch, pred, variates, quantiles)


def build_model(context_len, prediction_len, num_variates, quantiles):
    return _FallbackModel(context_len, prediction_len, num_variates, len(quantiles))


def build_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)
'''


def is_harness_task(challenge: dict) -> bool:
    """Always True — all tasks use the training harness."""
    return True


def validate(code: str, challenge: dict) -> tuple[bool, list[str]]:
    """Validate generated code. Returns (ok, list_of_errors)."""
    errors: list[str] = []

    # 0. Reject empty / whitespace-only code
    if not code or not code.strip():
        return False, ["Empty code — no source provided"]

    # 1. Syntax check
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, [f"SyntaxError: {exc}"]

    # 2. Forbidden imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    errors.append(f"Forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    errors.append(f"Forbidden import: {node.module}")

    # 3. Harness-task-specific checks
    if is_harness_task(challenge):
        func_defs = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = [a.arg for a in node.args.args]
                func_defs[node.name] = params

        for fname, required_params in HARNESS_REQUIRED.items():
            if fname not in func_defs:
                errors.append(f"Missing required function: {fname}")
            else:
                actual = func_defs[fname]
                # Skip 'self' if present
                actual_clean = [p for p in actual if p != "self"]
                for rp in required_params:
                    if rp not in actual_clean:
                        errors.append(
                            f"{fname} missing parameter: {rp}"
                        )

    # 4. FLOPs bounds check — only if no structural errors so far
    if not errors:
        flops_min, flops_max = extract_flops_budget(challenge)
        if flops_min or flops_max:
            gate_min = int(flops_min * 0.9)
            gate_max = int(flops_max * 1.1)
            estimated, err = estimate_flops(code, challenge)
            if err:
                errors.append(f"FLOPs estimation failed: {err}")
            elif estimated is not None:
                if estimated < gate_min:
                    errors.append(
                        f"Estimated FLOPs ({estimated:,}) below hard gate "
                        f"minimum ({gate_min:,}). Increase model capacity."
                    )
                elif estimated > gate_max:
                    errors.append(
                        f"Estimated FLOPs ({estimated:,}) above hard gate "
                        f"maximum ({gate_max:,}). Reduce model capacity."
                    )

    return len(errors) == 0, errors
