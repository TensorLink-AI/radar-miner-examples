"""Tool definitions and handlers for the OpenAI-SDK agent.

A deliberately minimal toolset — fewer tools means more focused reasoning
and fewer "explore everything once" loops. Start with: task analysis,
validation, FLOPs estimation, frontier inspection, submit. Add more
(scratchpad, papers, db queries) only when this version is confidently
production-quality.

Each tool entry follows the OpenAI function-calling schema. Handlers are
built per-round via :func:`build_handlers` so they can close over the
challenge dict.

A simple per-tool circuit breaker stops the LLM from calling the same
tool with the same failing args over and over.
"""
from __future__ import annotations

import json
import sys

from core.flops_estimator import estimate_flops
from core.validation import validate_code

# ── Schema (OpenAI function-calling format) ───────────────────────────

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "analyze_task",
            "description": (
                "Return the current challenge's task spec, parameters, "
                "constraints, objectives, and FLOPs budget as JSON."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_code",
            "description": (
                "Validate a candidate model. Returns 'ok' or 'errors: ...' "
                "covering syntax, missing build_model/build_optimizer, "
                "forbidden imports, FLOPs out of bucket, and output-shape "
                "mismatches. Always call this before `submit`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Full Python source to validate.",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_flops",
            "description": (
                "Run build_model and measure forward-pass FLOPs with the "
                "same FlopCounterMode the validator uses. Returns the "
                "FLOPs count or an error string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Full Python source containing build_model.",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_frontier",
            "description": (
                "List frontier members (just index + objectives). Use "
                "`get_frontier_member` to fetch a specific member's code."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_frontier_member",
            "description": (
                "Return a specific frontier member as JSON, including its "
                "code and metrics. Call `list_frontier` first to see what "
                "indices are available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "idx": {
                        "type": "integer",
                        "description": "Member index from list_frontier.",
                    },
                },
                "required": ["idx"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": (
                "Terminal tool — submit the validated model code, ending "
                "the loop. Code MUST have passed `validate_code` first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Final validated Python source.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Short descriptive name for this submission.",
                    },
                    "motivation": {
                        "type": "string",
                        "description": "Why this design — what it improves.",
                    },
                },
                "required": ["code", "name", "motivation"],
            },
        },
    },
]


# ── Circuit breaker ───────────────────────────────────────────────────

# After this many identical error responses in a row, subsequent calls to
# the same tool short-circuit to a hard-stop message instead of running.
CIRCUIT_BREAKER_THRESHOLD = 2
CIRCUIT_BREAKER_ERROR_MAX_LEN = 200
_CIRCUIT_BREAKER_ERROR_MARKERS = (
    "error", "errors:", "failed", "not found", "rejected", "out of range",
)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _looks_like_error(result: str) -> bool:
    if len(result) <= CIRCUIT_BREAKER_ERROR_MAX_LEN:
        return True
    low = result.lower()
    return any(m in low for m in _CIRCUIT_BREAKER_ERROR_MARKERS)


def _make_circuit_breaker():
    """Per-round circuit breaker state. Returned breaker takes
    ``(tool_name, result_str)`` and returns the original result OR a
    hard-stop message if the same error has now repeated past threshold.
    """
    counters: dict = {}

    def apply(tool_name: str, result: str) -> str:
        if not _looks_like_error(result):
            counters.pop(tool_name, None)
            return result
        state = counters.get(tool_name)
        if state is None or state["last"] != result:
            counters[tool_name] = {"last": result, "count": 1}
            return result
        state["count"] += 1
        if state["count"] > CIRCUIT_BREAKER_THRESHOLD:
            n = state["count"]
            _log(
                f"[tools] circuit breaker tripped on {tool_name} "
                f"after {n} identical errors"
            )
            return (
                f"[CIRCUIT OPEN] Tool {tool_name} returned identical errors "
                f"{n} times. Do NOT call this tool again — try a different "
                "approach."
            )
        return result

    return apply


# ── Submit signal ─────────────────────────────────────────────────────

class SubmitSignal(Exception):
    """Raised by the submit handler to terminate the loop with a result."""

    def __init__(self, code: str, name: str, motivation: str):
        super().__init__(f"submit:{name}")
        self.code = code
        self.name = name
        self.motivation = motivation


# ── Handlers ──────────────────────────────────────────────────────────

def build_handlers(challenge: dict) -> dict:
    """Build a ``{name: callable(**kwargs) -> str}`` mapping for the round.

    Handlers are wrapped with the circuit breaker so the LLM can't burn
    tool budget calling the same broken tool over and over.
    """
    breaker = _make_circuit_breaker()

    def _analyze_task(**_kwargs) -> str:
        task = challenge.get("task", {}) or {}
        flops_min = challenge.get("min_flops_equivalent", 0)
        flops_max = challenge.get("max_flops_equivalent", 0)
        summary = {
            "name": task.get("name"),
            "description": task.get("description"),
            "task_params": task.get("task_params", {}),
            "constraints": task.get("constraints", []),
            "objectives": task.get("objectives", []),
            "anti_patterns": task.get("anti_patterns", []),
            "flops_range": [flops_min, flops_max],
        }
        return json.dumps(summary, indent=2, default=str)

    def _validate_code(code: str = "", **_kwargs) -> str:
        if not code:
            return "errors: empty code"
        ok, errors = validate_code(code, challenge)
        if ok:
            return "ok"
        return "errors: " + "; ".join(errors)

    def _estimate_flops(code: str = "", **_kwargs) -> str:
        if not code:
            return "error: empty code"
        flops, err = estimate_flops(code, challenge)
        if err:
            return f"error: {err}"
        return f"estimated_flops: {flops:,}"

    def _list_frontier(**_kwargs) -> str:
        frontier = challenge.get("feasible_frontier") or []
        if not frontier:
            return "frontier is empty (bootstrapping round)"
        items = [
            {
                "idx": i,
                "name": f.get("name", "?"),
                "objectives": f.get("objectives", {}),
            }
            for i, f in enumerate(frontier)
        ]
        return json.dumps(items, indent=2)

    def _get_frontier_member(idx: int = -1, **_kwargs) -> str:
        frontier = challenge.get("feasible_frontier") or []
        if not frontier:
            return "error: frontier is empty"
        if not isinstance(idx, int) or idx < 0 or idx >= len(frontier):
            return f"error: idx {idx} out of range [0, {len(frontier)})"
        return json.dumps(frontier[idx], indent=2, default=str)

    def _submit(code: str = "", name: str = "", motivation: str = "",
                **_kwargs) -> str:
        # Terminal tool — raise to unwind the tool loop with the result.
        raise SubmitSignal(code=code, name=name, motivation=motivation)

    raw = {
        "analyze_task": _analyze_task,
        "validate_code": _validate_code,
        "estimate_flops": _estimate_flops,
        "list_frontier": _list_frontier,
        "get_frontier_member": _get_frontier_member,
        "submit": _submit,
    }

    def _wrap(name, fn):
        def wrapped(**kwargs):
            # SubmitSignal must propagate untouched — do not pass it
            # through the breaker.
            try:
                result = fn(**kwargs)
            except SubmitSignal:
                raise
            except Exception as exc:
                result = f"error: {exc}"
            return breaker(name, str(result))
        return wrapped

    return {name: _wrap(name, fn) for name, fn in raw.items()}
