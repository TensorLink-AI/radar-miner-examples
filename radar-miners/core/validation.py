"""AST-based code validation — syntax, required functions, forbidden imports."""

import ast
import sys

FORBIDDEN_IMPORTS = {"subprocess", "socket", "ftplib"}

HARNESS_REQUIRED = {
    "build_model": ["context_len", "prediction_len", "num_variates", "quantiles"],
    "build_optimizer": ["model"],
}

HARNESS_OPTIONAL = [
    "training_config", "init_weights", "configure_amp",
    "transform_batch", "on_step_end", "build_scheduler", "compute_loss",
]


def is_harness_task(challenge: dict) -> bool:
    """Check if the task uses the training harness (vs standalone)."""
    run_cmd = challenge.get("task", {}).get("run_command", "")
    return "harness.py" in run_cmd


def validate(code: str, challenge: dict) -> tuple[bool, list[str]]:
    """Validate generated code. Returns (ok, list_of_errors)."""
    errors: list[str] = []

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

    return len(errors) == 0, errors
