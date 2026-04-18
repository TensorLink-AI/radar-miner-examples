"""Tool definitions and handlers for the OpenAI-SDK agent.

Matches the autonomous agent's tool surface: research (arxiv + DB),
analysis (task + FLOPs + layer probes), design (sketch), validation
(validate / trace / check output), state (scratchpad read/write),
control (submit, time_remaining).

Each tool entry follows the OpenAI function-calling schema. Handlers are
built per-round via :func:`build_handlers` so they close over the round's
challenge dict, HTTP client, scratch directory, and deadline. When a
dependency is missing (e.g. the client is ``None`` during unit tests),
the affected handler returns a short "unavailable" string rather than
raising.

A simple per-tool circuit breaker stops the LLM from calling the same
tool with the same failing args over and over.
"""
from __future__ import annotations

import copy
import json
import sys
import time

import torch
import torch.nn as nn

from core import call_with_timeout
from core.arch_knowledge import scan_frontier_ops
from core.flops_estimator import estimate_flops, suggest_resize
from core.history import (
    extract_flops_budget, format_history, get_history, load_state, save_state,
)
from core.input_shape import infer_input
from core.output_shape import infer_output_shape, verify_output_shape
from core.trace import format_trace, trace_architecture
from core.validation import validate_code

TOOL_HTTP_TIMEOUT = 15       # seconds per research/DB request
FRONTIER_HTTP_TIMEOUT = 10   # seconds for frontier fetch

# ── Schema (OpenAI function-calling format) ───────────────────────────

TOOLS: list[dict] = [
    # ── Research ─────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": (
                "Search arxiv for papers relevant to your architecture "
                "design. Returns titles and abstracts. Use this to find "
                "state-of-the-art techniques for the task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query (e.g. 'efficient <task_name> "
                            "neural network')"
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results (default 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_db",
            "description": (
                "Query the experiment database. Known endpoints:\n"
                "  GET  /experiments/recent?n=15\n"
                "  GET  /experiments/failures?n=5\n"
                "  GET  /provenance/component_stats\n"
                "  GET  /provenance/dead_ends\n"
                "  GET  /experiments/{id}\n"
                "  POST /experiments/search\n"
                "Other paths may also work — explore what's available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST"],
                        "description": "HTTP method (default GET)",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "API path, e.g. '/experiments/recent?n=10'"
                        ),
                    },
                    "body": {
                        "type": "object",
                        "description": "JSON body for POST (optional)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    # ── Analysis ─────────────────────────────────────────────────
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
            "name": "estimate_layer_flops",
            "description": (
                "Instantiate a single layer / small module and measure its "
                "forward-pass FLOPs with FlopCounterMode. Provide code that "
                "assigns the module to a variable called `layer` and an "
                "input shape. Works with ANY PyTorch module."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python code that defines and instantiates a "
                            "module. Must assign to `layer`. Example: "
                            "'import torch.nn as nn\\nlayer = nn.Conv1d(7, 32, 5)'"
                        ),
                    },
                    "input_shape": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "Input tensor shape incl. batch dim, e.g. [1, 512, 7]"
                        ),
                    },
                },
                "required": ["code", "input_shape"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sketch_architecture",
            "description": (
                "Test a complete architecture sketch by supplying only "
                "build_model. The tool runs it, measures total FLOPs, "
                "traces per-layer shapes, and checks output shape against "
                "the task constraint. build_optimizer is auto-added if "
                "missing. Use this to validate a design before writing "
                "full production code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python code containing a `def build_model(...)` "
                            "that returns an nn.Module. No need to include "
                            "build_optimizer — one is added automatically."
                        ),
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
                        "description": (
                            "Full Python source containing build_model."
                        ),
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
    # ── Validation ───────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "trace_architecture",
            "description": (
                "Run a dummy forward pass through build_model and return "
                "an op-by-op trace: each leaf module's name, class, input "
                "shape, output shape, and parameter count. Memory-efficient "
                "(only shape tuples and ints are captured)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Full Python source containing build_model."
                        ),
                    },
                    "max_rows": {
                        "type": "integer",
                        "description": (
                            "Max trace rows to include in the rendered "
                            "output (default 60)."
                        ),
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_output_shape",
            "description": (
                "Run build_model and a dummy forward pass, then compare "
                "the output tensor's shape against the expected shape "
                "parsed from the task constraints. Catches tensor-size "
                "mismatches before submission."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Full Python source containing build_model."
                        ),
                    },
                },
                "required": ["code"],
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
    # ── State ────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "read_scratchpad",
            "description": (
                "Load persistent state from scratchpad. Contains history "
                "of your previous submissions across rounds — what you "
                "tried, what worked, what didn't. Check this first each "
                "round."
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
            "name": "write_scratchpad",
            "description": (
                "Save notes to the persistent scratchpad for future "
                "rounds. Use this to record what you learned, what "
                "worked, and what to try next time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "notes": {
                        "type": "string",
                        "description": (
                            "Free-form notes to save for future rounds."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    # ── Control ──────────────────────────────────────────────────
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
                        "description": (
                            "Short descriptive name for this submission."
                        ),
                    },
                    "motivation": {
                        "type": "string",
                        "description": (
                            "Why this design — what it improves."
                        ),
                    },
                },
                "required": ["code", "name", "motivation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "time_remaining",
            "description": (
                "Check how many seconds remain in the time budget. If "
                "time is low, submit what you have rather than starting "
                "new research."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
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
    "unavailable", "timeout", "timed out",
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

def build_handlers(
    challenge: dict,
    *,
    client=None,
    scratch_dir: str | None = None,
    deadline: float | None = None,
) -> dict:
    """Build a ``{name: callable(**kwargs) -> str}`` mapping for the round.

    ``client`` is a ``GatedClient``-compatible object (has
    ``get_json``/``post_json``) used for paper search and DB queries.
    ``scratch_dir`` is a filesystem path where scratchpad state lives.
    ``deadline`` is the monotonic wall-clock the round must finish by;
    ``time_remaining`` reports against it.

    Any of these may be ``None`` — the affected tools return an
    "unavailable" string instead of raising. Handlers are wrapped with
    the circuit breaker so the LLM can't burn tool budget calling the
    same broken tool over and over.
    """
    breaker = _make_circuit_breaker()

    db_url = (challenge.get("db_url") or "").rstrip("/")
    desearch_url = (challenge.get("desearch_url") or "").rstrip("/")
    flops_min, flops_max = extract_flops_budget(challenge)
    target_flops = int(flops_max * 0.6) if flops_max else 0

    state_holder = {
        "state": load_state(scratch_dir) if scratch_dir else {}
    }

    # Frontier fetched from the challenge dict only — the openai_sdk
    # agent has historically treated ``feasible_frontier`` as canonical
    # and does not hit a /frontier endpoint.
    def _frontier() -> list:
        frontier = challenge.get("feasible_frontier") or []
        if not frontier:
            frontier = challenge.get("pareto_frontier") or []
        return frontier if isinstance(frontier, list) else []

    def _fmt(data) -> str:
        try:
            return json.dumps(data, default=str, indent=2)
        except (TypeError, ValueError):
            return str(data)

    # ── Research ─────────────────────────────────────────────────

    def _search_papers(query: str = "", max_results: int = 5,
                       **_kwargs) -> str:
        if not query:
            return "error: empty query"
        if client is None or not desearch_url:
            return "paper search unavailable (no client or desearch_url)"
        try:
            resp = call_with_timeout(
                client.post_json,
                args=(f"{desearch_url}/search",
                      {"query": query, "max_results": max_results}),
                timeout=TOOL_HTTP_TIMEOUT,
            )
            results = resp.get("results", []) if isinstance(resp, dict) else []
            if not results:
                return "no papers found"
            lines = []
            for r in results[:max_results]:
                lines.append(f"**{r.get('title', 'untitled')}**")
                abstract = r.get("abstract", "")
                if abstract:
                    lines.append(abstract[:500])
                lines.append("")
            return "\n".join(lines)
        except TimeoutError:
            return f"paper search timed out after {TOOL_HTTP_TIMEOUT}s"
        except Exception as exc:
            return f"paper search failed: {exc}"

    def _query_db(path: str = "", method: str = "GET",
                  body: dict | None = None, **_kwargs) -> str:
        if not path:
            return "error: empty path"
        if client is None or not db_url:
            return "experiment DB unavailable (no client or db_url)"
        if not path.startswith("/"):
            path = "/" + path
        url = f"{db_url}{path}"
        try:
            if method.upper() == "POST":
                result = call_with_timeout(
                    client.post_json, args=(url, body or {}),
                    timeout=TOOL_HTTP_TIMEOUT,
                )
            else:
                result = call_with_timeout(
                    client.get_json, args=(url,),
                    timeout=TOOL_HTTP_TIMEOUT,
                )
            if isinstance(result, dict) and "error" in result:
                return f"DB returned error: {result['error']}"
            formatted = _fmt(result)
            if len(formatted) > 8000:
                formatted = (
                    formatted[:8000]
                    + "\n... (truncated, try a more specific query)"
                )
            return formatted
        except TimeoutError:
            return f"DB query timed out after {TOOL_HTTP_TIMEOUT}s"
        except Exception as exc:
            return f"DB query failed: {exc}"

    # ── Analysis ─────────────────────────────────────────────────

    def _analyze_task(**_kwargs) -> str:
        task = challenge.get("task", {}) or {}
        summary = {
            "name": task.get("name"),
            "description": task.get("description"),
            "task_params": task.get("task_params", {}),
            "constraints": task.get("constraints", []),
            "objectives": task.get("objectives", []),
            "anti_patterns": task.get("anti_patterns", []),
            "flops_range": [flops_min, flops_max],
        }
        frontier = _frontier()
        if frontier:
            summary["frontier_ops"] = scan_frontier_ops(frontier)
        return json.dumps(summary, indent=2, default=str)

    def _estimate_layer_flops(code: str = "",
                              input_shape: list | None = None,
                              **_kwargs) -> str:
        if not code or not code.strip():
            return "error: empty code"
        if not input_shape:
            return "error: input_shape is required"
        try:
            from torch.utils.flop_counter import FlopCounterMode
        except ImportError:
            return "error: FlopCounterMode not available in this torch version"

        namespace: dict = {"torch": torch, "nn": nn}
        try:
            exec(compile(code, "<layer_snippet>", "exec"), namespace)
        except Exception as exc:
            return f"layer code execution failed: {exc}"

        layer = namespace.get("layer")
        if layer is None:
            return (
                "error: snippet did not assign a variable named `layer`. "
                "Example: `layer = nn.Conv1d(7, 32, 5)`."
            )
        if not isinstance(layer, nn.Module):
            return f"error: `layer` is not an nn.Module (got {type(layer).__name__})"

        shape = [int(d) for d in input_shape]
        try:
            dummy = torch.randn(*shape)
        except Exception as exc:
            return f"failed to build dummy input of shape {shape}: {exc}"

        try:
            layer.eval()
            with torch.no_grad():
                counter = FlopCounterMode(display=False)
                with counter:
                    out = layer(dummy)
                total = int(counter.get_total_flops())
        except Exception as exc:
            return f"forward pass failed: {exc}"

        out_shape = (
            tuple(out.shape) if isinstance(out, torch.Tensor) else "(non-tensor)"
        )
        param_count = sum(p.numel() for p in layer.parameters())
        lines = [
            f"measured_flops: {total:,}",
            f"output_shape: {out_shape}",
            f"params: {param_count:,}",
        ]
        if target_flops:
            pct = 100.0 * total / target_flops
            lines.append(
                f"budget_usage: {pct:.1f}% of target {target_flops:,}"
            )
        return "\n".join(lines)

    def _sketch_architecture(code: str = "", **_kwargs) -> str:
        if not code or not code.strip():
            return "error: empty code"

        if "def build_optimizer" not in code:
            code = code.rstrip() + (
                "\n\ndef build_optimizer(model):\n"
                "    import torch\n"
                "    return torch.optim.Adam(model.parameters(), lr=1e-3)\n"
            )

        sink: list = []
        estimated, err = estimate_flops(code, challenge, sink)
        if err:
            return f"sketch forward pass failed: {err}"

        lines: list[str] = []
        entries, trace_err = trace_architecture(code, challenge)
        if trace_err:
            lines.append(f"(per-layer trace unavailable: {trace_err})")
        elif entries:
            lines.append(format_trace(entries, max_rows=40))
        else:
            lines.append("(no leaf-op entries captured)")

        lines.append("")
        if estimated is not None:
            pct = (100.0 * estimated / target_flops) if target_flops else 0.0
            lines.append(
                f"TOTAL FLOPs: {estimated:,}"
                + (
                    f" ({pct:.0f}% of target {target_flops:,})"
                    if target_flops else ""
                )
            )
            if flops_min and flops_max:
                gate_min = int(flops_min * 0.9)
                gate_max = int(flops_max * 1.1)
                if estimated < gate_min:
                    lines.append(
                        f"STATUS: below hard gate [{gate_min:,}, "
                        f"{gate_max:,}] — increase capacity"
                    )
                    hint = suggest_resize(
                        estimated, gate_min, gate_max, target_flops,
                    )
                    if hint:
                        lines.append(hint)
                elif estimated > gate_max:
                    lines.append(
                        f"STATUS: above hard gate [{gate_min:,}, "
                        f"{gate_max:,}] — reduce capacity"
                    )
                    hint = suggest_resize(
                        estimated, gate_min, gate_max, target_flops,
                    )
                    if hint:
                        lines.append(hint)
                else:
                    lines.append(
                        f"STATUS: within hard gate [{gate_min:,}, "
                        f"{gate_max:,}]"
                    )

        task = challenge.get("task", {}) or {}
        tp = task.get("task_params", {}) or {}
        constraints = task.get("constraints", []) or []
        expected = infer_output_shape(tp, constraints)
        if expected is not None and sink:
            shape_err = verify_output_shape(sink[0], expected)
            actual = tuple(sink[0])
            if shape_err:
                lines.append(
                    f"OUTPUT SHAPE: {actual} — MISMATCH: {shape_err}"
                )
            else:
                lines.append(
                    f"OUTPUT SHAPE: {actual} — matches expected"
                )
        elif sink:
            lines.append(
                f"OUTPUT SHAPE: {tuple(sink[0])} (no declared expectation)"
            )

        lines.append(
            "\nNote: sketch_architecture does NOT reject on FLOPs bounds. "
            "Use validate_code for the final pre-submission check."
        )
        return "\n".join(lines)

    def _estimate_flops(code: str = "", **_kwargs) -> str:
        if not code:
            return "error: empty code"
        flops, err = estimate_flops(code, challenge)
        if err:
            return f"error: {err}"
        return f"estimated_flops: {flops:,}"

    def _list_frontier(**_kwargs) -> str:
        frontier = _frontier()
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
        frontier = _frontier()
        if not frontier:
            return "error: frontier is empty"
        if not isinstance(idx, int) or idx < 0 or idx >= len(frontier):
            return f"error: idx {idx} out of range [0, {len(frontier)})"
        return json.dumps(frontier[idx], indent=2, default=str)

    # ── Validation ───────────────────────────────────────────────

    def _trace_architecture(code: str = "", max_rows: int = 60,
                            **_kwargs) -> str:
        if not code or not code.strip():
            return "error: empty code"
        entries, err = trace_architecture(code, challenge)
        if err:
            return f"trace failed: {err}"
        if not entries:
            return (
                "trace produced no entries — the model has no leaf modules "
                "with parameters, or the forward pass produced no tensor "
                "outputs through hooked modules."
            )
        return format_trace(entries, max_rows=max_rows)

    def _check_output_shape(code: str = "", **_kwargs) -> str:
        if not code or not code.strip():
            return "error: empty code"
        task = challenge.get("task", {}) or {}
        tp = task.get("task_params", {}) or {}
        constraints = task.get("constraints", []) or []
        expected = infer_output_shape(tp, constraints)
        if expected is None:
            return (
                "no parseable output-shape constraint in task.constraints; "
                "skipping shape check"
            )
        sink: list = []
        _, err = estimate_flops(code, challenge, sink)
        if err:
            return f"could not run forward pass: {err}"
        if not sink:
            return (
                "forward pass completed but no tensor output was captured. "
                "Add a concrete forward() that returns a torch.Tensor."
            )
        shape_err = verify_output_shape(sink[0], expected)
        if shape_err:
            return f"FAILED: {shape_err}"
        expected_pretty = ", ".join(
            str(e) if e >= 0 else "?" for e in expected
        )
        return (
            f"PASSED: output shape {tuple(sink[0])} matches expected "
            f"(B, {expected_pretty})."
        )

    def _validate_code(code: str = "", **_kwargs) -> str:
        if not code:
            return "errors: empty code"
        ok, errors = validate_code(code, challenge)
        if ok:
            return "ok"
        return "errors: " + "; ".join(errors)

    # ── State ────────────────────────────────────────────────────

    def _read_scratchpad(**_kwargs) -> str:
        state = state_holder["state"]
        history_entries = get_history(state)
        notes = state.get("agent_notes", "")
        parts = []
        if notes:
            parts.append(f"## Saved Notes\n{notes}")
        if history_entries:
            parts.append(
                "## Submission History\n"
                + format_history(history_entries, max_entries=10)
            )
        if not parts:
            return "scratchpad is empty — this is your first round"
        return "\n\n".join(parts)

    def _write_scratchpad(notes: str = "", **_kwargs) -> str:
        state = state_holder["state"]
        if notes:
            state["agent_notes"] = notes
        state_holder["state"] = state
        return "scratchpad updated"

    # ── Control ──────────────────────────────────────────────────

    def _submit(code: str = "", name: str = "", motivation: str = "",
                **_kwargs) -> str:
        raise SubmitSignal(code=code, name=name, motivation=motivation)

    def _time_remaining(**_kwargs) -> str:
        if deadline is None:
            return "no deadline set"
        remaining = max(0.0, deadline - time.monotonic())
        if remaining > 60:
            return (
                f"{remaining:.0f}s remaining "
                f"({remaining / 60:.1f} minutes)"
            )
        if remaining > 0:
            return f"{remaining:.0f}s remaining — wrap up soon"
        return "time is up — submit now or the round ends without a submission"

    raw = {
        "search_papers": _search_papers,
        "query_db": _query_db,
        "analyze_task": _analyze_task,
        "estimate_layer_flops": _estimate_layer_flops,
        "sketch_architecture": _sketch_architecture,
        "estimate_flops": _estimate_flops,
        "list_frontier": _list_frontier,
        "get_frontier_member": _get_frontier_member,
        "trace_architecture": _trace_architecture,
        "check_output_shape": _check_output_shape,
        "validate_code": _validate_code,
        "read_scratchpad": _read_scratchpad,
        "write_scratchpad": _write_scratchpad,
        "submit": _submit,
        "time_remaining": _time_remaining,
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

        # Expose state_holder on the submit handler so the agent can
        # persist scratchpad notes after the loop.
        if name == "submit":
            wrapped._state_holder = state_holder  # type: ignore[attr-defined]
        return wrapped

    return {name: _wrap(name, fn) for name, fn in raw.items()}


def build_tools(challenge: dict | None = None) -> list[dict]:
    """Return a tools list customized for this challenge.

    Swaps the placeholder in the ``search_papers`` description for the
    task's actual name so the LLM sees a relevant example query.
    """
    if not challenge:
        return TOOLS
    task = challenge.get("task", {}) or {}
    task_name = task.get("name", "") or "neural network"
    tools = copy.deepcopy(TOOLS)
    for t in tools:
        fn = t.get("function", {})
        if fn.get("name") == "search_papers":
            props = fn.get("parameters", {}).get("properties", {})
            q = props.get("query", {})
            q["description"] = (
                f"Search query (e.g. 'efficient {task_name} neural network')"
            )
    return tools
