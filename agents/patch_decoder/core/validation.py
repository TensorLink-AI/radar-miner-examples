"""AST-based pre-flight validation — mirrors the validator's checks exactly."""

import ast

FORBIDDEN_IMPORTS = {"subprocess", "socket", "ftplib"}

REQUIRED_FUNCTIONS = {
    "build_model": ["context_len", "prediction_len", "num_variates", "quantiles"],
    "build_optimizer": ["model"],
}


def validate_code(code: str) -> tuple[bool, list[str]]:
    """Validate generated code against the validator's requirements.

    Returns (ok, list_of_errors).
    """
    errors: list[str] = []

    if not code or not code.strip():
        return False, ["Empty code — no source provided"]

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, [f"SyntaxError: {exc}"]

    top_level_funcs: dict[str, list[str]] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = [a.arg for a in node.args.args if a.arg != "self"]
            top_level_funcs[node.name] = params

    for fname, required_params in REQUIRED_FUNCTIONS.items():
        if fname not in top_level_funcs:
            errors.append(f"Missing required top-level function: {fname}")
        else:
            actual = top_level_funcs[fname]
            for rp in required_params:
                if rp not in actual:
                    errors.append(f"{fname} missing parameter: {rp}")

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

    return len(errors) == 0, errors
