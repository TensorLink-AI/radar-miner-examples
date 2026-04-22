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
import os
import sys
import time

import torch
import torch.nn as nn

from core import call_with_timeout
from core.arch_knowledge import scan_frontier_ops
from core.flops_estimator import estimate_flops, suggest_resize
from core.history import (
    add_note, extract_flops_budget, format_history, format_notes,
    get_history, load_state, save_state,
)
from core.input_shape import infer_input
from core.output_shape import infer_output_shape, verify_output_shape
from core.sizing import MAX_PROBES, sweep_sizes
from core.trace import format_trace, trace_architecture
from core.validation import validate_code

TOOL_HTTP_TIMEOUT = 15       # seconds per research/DB request
FRONTIER_HTTP_TIMEOUT = 10   # seconds for frontier fetch

# Scratchpad file tool limits. The scratchpad upload itself is capped by
# ``challenge["scratchpad_max_mb"]``; these caps keep a single round from
# burning the whole budget on one file.
FILE_MAX_BYTES = 256 * 1024           # 256 KiB per file
FILE_MAX_COUNT = 50                   # total user-written files
FILE_SEARCH_MAX_MATCHES_PER_FILE = 5  # keep hits-per-file output bounded
FILE_SEARCH_MAX_TOTAL_MATCHES = 30    # global cap across all files
# ``state.json`` is owned by the history module and the ``_scratchpad``
# notes file is owned by write_scratchpad — don't let the LLM clobber
# either from the generic file tools.
RESERVED_FILE_NAMES = frozenset({"state.json"})

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
                "Query the experiment database. Available endpoints:\n"
                "  GET  /frontier                     — feasible frontier for the current round\n"
                "  GET  /experiments/recent?n=20      — recent experiments (default n=20)\n"
                "  GET  /experiments/pareto           — Pareto-optimal experiments by task\n"
                "  GET  /experiments/{index}          — single experiment by id\n"
                "  GET  /experiments/stats            — DB statistics\n"
                "  POST /experiments/search           — search experiments by query\n"
                "  GET  /experiments/lineage/{index}  — experiment ancestry\n"
                "  GET  /experiments/diff/{a}/{b}     — diff two experiments\n"
                "These are independent views of the same data — pick whichever "
                "fits what you're investigating. Other paths may also work; "
                "feel free to explore."
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
    {
        "type": "function",
        "function": {
            "name": "size_to_flops",
            "description": (
                "Sweep a scalar size knob to land on a target FLOPs "
                "count. Provide code containing a `{{SIZE}}` "
                "placeholder (any integer knob — hidden_dim, num_layers, "
                "channels, etc.) and the range to search. Runs up to "
                f"{MAX_PROBES} measurements using geometric-probe-then-"
                "refine and returns the MEASURED (size, flops) pair "
                "closest to the target. Use this BEFORE calling "
                "validate_code — it saves FLOPs-bound iterations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code_template": {
                        "type": "string",
                        "description": (
                            "Python source containing `build_model` "
                            "with a `{{SIZE}}` placeholder. The "
                            "placeholder is substituted with each "
                            "probe's integer. Example: "
                            "`hidden = {{SIZE}}` inside build_model."
                        ),
                    },
                    "size_min": {
                        "type": "integer",
                        "description": (
                            "Smallest size to probe (must be > 0)."
                        ),
                    },
                    "size_max": {
                        "type": "integer",
                        "description": (
                            "Largest size to probe (must be >= size_min)."
                        ),
                    },
                    "target_flops": {
                        "type": "integer",
                        "description": (
                            "Target FLOPs count. Defaults to 60% of the "
                            "challenge's max_flops_equivalent when 0 or "
                            "omitted."
                        ),
                    },
                },
                "required": ["code_template", "size_min", "size_max"],
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
                "Record a structured note for future rounds. Pass ONE "
                "of: `hypothesis` (something to test next), `dead_end` "
                "+ `reason` (an approach that failed and why), or "
                "`observation` (a task-agnostic fact learned). Each "
                "section is capped at 20 entries (oldest is dropped). "
                "You MUST write at least one note before calling "
                "`submit`. The deprecated free-form `notes` field is "
                "still accepted but prefer the structured fields."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hypothesis": {
                        "type": "string",
                        "description": (
                            "Appended to open_hypotheses. Example: "
                            "'Try depthwise-sep convs for the same "
                            "receptive field at lower FLOPs.'"
                        ),
                    },
                    "dead_end": {
                        "type": "string",
                        "description": (
                            "Short name of an approach that failed. "
                            "Pair with `reason`."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why the dead_end failed.",
                    },
                    "observation": {
                        "type": "string",
                        "description": (
                            "A task-agnostic fact learned this round "
                            "(e.g. 'output shape inferred as (B, N, K)'). "
                            "Appended to task_observations."
                        ),
                    },
                    "notes": {
                        "type": "string",
                        "description": (
                            "(DEPRECATED) Free-form notes string. "
                            "Prefer hypothesis / dead_end / observation."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List files you have written to the scratchpad directory "
                "in previous rounds. Good for keeping structured markdown "
                "notes (design.md, task-notes.md, etc.) alongside the "
                "free-form scratchpad string."
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
            "name": "read_file",
            "description": (
                "Read a file you previously wrote to the scratchpad "
                "directory. Use `list_files` first to see what's there."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "File name (basename only, no slashes or "
                            "parent references). Example: 'design.md'."
                        ),
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write a text file to the scratchpad directory (persists "
                "across rounds). Use for markdown notes, plans, or any "
                "structured text you want to keep. Overwrites if the file "
                "exists."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "File name (basename only, no slashes or "
                            "parent references). Example: 'design.md'."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file contents as text.",
                    },
                },
                "required": ["name", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": (
                "Case-insensitive substring search across all scratchpad "
                "files. Returns matching file names and the surrounding "
                "lines with line numbers. Use to find notes on a topic "
                "without rereading every file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Substring to search for (case-insensitive)."
                        ),
                    },
                },
                "required": ["query"],
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
    state: dict | None = None,
) -> dict:
    """Build a ``{name: callable(**kwargs) -> str}`` mapping for the round.

    ``client`` is a ``GatedClient``-compatible object (has
    ``get_json``/``post_json``) used for paper search and DB queries.
    ``scratch_dir`` is a filesystem path where scratchpad state lives.
    ``deadline`` is the monotonic wall-clock the round must finish by;
    ``time_remaining`` reports against it. ``state``, when provided,
    overrides the on-disk load — used by callers that need to merge
    round-result feedback before the handlers see the state.

    Any of these may be ``None`` — the affected tools return an
    "unavailable" string instead of raising. Handlers are wrapped with
    the circuit breaker so the LLM can't burn tool budget calling the
    same broken tool over and over.
    """
    breaker = _make_circuit_breaker()
    # Per-round telemetry — every wrapped handler bumps this. The agent
    # reads it at end-of-round to log a usage summary. Handlers see it
    # via closure only; nothing inside a handler should mutate it.
    _call_counts: dict[str, int] = {}

    db_url = (challenge.get("db_url") or "").rstrip("/")
    desearch_url = (challenge.get("desearch_url") or "").rstrip("/")
    flops_min, flops_max = extract_flops_budget(challenge)
    target_flops = int(flops_max * 0.6) if flops_max else 0

    if state is None:
        state = load_state(scratch_dir) if scratch_dir else {}
    # ``wrote_this_round`` feeds the submit-time nag; ``submit_nag_count``
    # makes the nag a one-shot so a determined LLM isn't blocked.
    state_holder: dict = {
        "state": state,
        "wrote_this_round": False,
        "submit_nag_count": 0,
    }

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

    def _size_to_flops(code_template: str = "", size_min: int = 0,
                       size_max: int = 0, target_flops: int = 0,
                       **_kwargs) -> str:
        if not code_template or not code_template.strip():
            return "error: empty code_template"
        if "{{SIZE}}" not in code_template:
            return (
                "error: code_template must contain a `{{SIZE}}` "
                "placeholder (the integer knob to sweep)"
            )
        try:
            size_min = int(size_min)
            size_max = int(size_max)
            target_flops = int(target_flops or 0)
        except (TypeError, ValueError):
            return "error: size_min/size_max/target_flops must be integers"
        if size_min <= 0:
            return f"error: size_min must be > 0 (got {size_min})"
        if size_max < size_min:
            return (
                f"error: size_max ({size_max}) < size_min ({size_min})"
            )
        if target_flops <= 0:
            target_flops = int(flops_max * 0.6) if flops_max else 0
        if target_flops <= 0:
            return (
                "error: no target_flops supplied and the challenge has "
                "no FLOPs budget to default from"
            )

        def _probe(size: int) -> int | None:
            code = code_template.replace("{{SIZE}}", str(size))
            flops, err = estimate_flops(code, challenge)
            if err or flops is None:
                return None
            return int(flops)

        result = sweep_sizes(
            _probe, size_min, size_max, target_flops,
        )
        probes = result["probes"]
        failures = result["failures"]
        best = result["best"]
        n_probes = result["n_probes"]

        lines = [
            f"target_flops: {target_flops:,}",
            f"search_range: [{size_min:,}, {size_max:,}]",
            f"probes: {n_probes} ({len(probes)} ok, {len(failures)} failed)",
        ]
        if best is None:
            lines.append(
                "best: NONE — every probe failed. Check that "
                "code_template defines build_model and the SIZE knob is "
                "wired to a real shape dimension."
            )
            if failures:
                lines.append(
                    f"failing sizes (sample): {failures[:5]}"
                )
        else:
            size, flops = best
            pct = 100.0 * flops / target_flops
            lines.append(
                f"best: size={size:,} flops={flops:,} "
                f"({pct:.1f}% of target)"
            )
            if flops_min and flops_max:
                gate_min = int(flops_min * 0.9)
                gate_max = int(flops_max * 1.1)
                if gate_min <= flops <= gate_max:
                    lines.append(
                        f"status: within hard gate "
                        f"[{gate_min:,}, {gate_max:,}]"
                    )
                else:
                    lines.append(
                        f"status: OUTSIDE hard gate "
                        f"[{gate_min:,}, {gate_max:,}] — widen the "
                        "search range or change the knob"
                    )
            if len(probes) > 1:
                sample = ", ".join(
                    f"{s}:{f:,}" for s, f in probes[:6]
                )
                lines.append(f"probe_samples: {sample}")
        return "\n".join(lines)

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
        legacy_notes = state.get("agent_notes", "")
        score_direction = (
            challenge.get("score_direction") or "minimize"
        )
        parts = []
        structured = format_notes(state)
        if structured:
            parts.append(structured)
        if legacy_notes:
            parts.append(f"## Free-form Notes (deprecated)\n{legacy_notes}")
        if history_entries:
            parts.append(
                "## Submission History\n"
                + format_history(
                    history_entries,
                    max_entries=10,
                    score_direction=score_direction,
                )
            )
        if not parts:
            return "scratchpad is empty — this is your first round"
        return "\n\n".join(parts)

    def _write_scratchpad(hypothesis: str = "", dead_end: str = "",
                          reason: str = "", observation: str = "",
                          notes: str = "", **_kwargs) -> str:
        state = state_holder["state"]
        wrote: list[str] = []
        if hypothesis and hypothesis.strip():
            add_note(state, "open_hypotheses", hypothesis)
            wrote.append("hypothesis")
        if dead_end and dead_end.strip():
            combined = (
                f"{dead_end.strip()} — {reason.strip()}"
                if reason and reason.strip()
                else dead_end.strip()
            )
            add_note(state, "dead_ends", combined)
            wrote.append("dead_end")
        if observation and observation.strip():
            add_note(state, "task_observations", observation)
            wrote.append("observation")
        if notes and notes.strip():
            # Deprecated path — still honored so in-flight rounds don't
            # crash. Stash alongside the structured notes and warn.
            state["agent_notes"] = notes.strip()
            wrote.append("notes(deprecated)")
            _log(
                "[tools] write_scratchpad(notes=...) is deprecated — use "
                "hypothesis/dead_end/observation instead"
            )
        if not wrote:
            return (
                "error: nothing to write — pass at least one of "
                "hypothesis, dead_end (+reason), observation, or notes"
            )
        state_holder["state"] = state
        state_holder["wrote_this_round"] = True
        return f"scratchpad updated ({', '.join(wrote)})"

    # ── Files (scratchpad directory) ─────────────────────────────

    def _safe_file_path(name: str) -> tuple[str | None, str | None]:
        """Resolve ``name`` to an absolute path inside ``scratch_dir``.

        Returns ``(path, None)`` on success or ``(None, error)`` when the
        name is unsafe or the scratch directory is unavailable. Only
        simple basenames are allowed — no slashes, no parent refs, no
        absolute paths. Reserved names are rejected.
        """
        if scratch_dir is None:
            return None, "files unavailable (no scratchpad directory)"
        if not name or not isinstance(name, str):
            return None, "error: name is required"
        if name != os.path.basename(name) or name in (".", ".."):
            return None, (
                f"error: {name!r} is not a plain filename "
                "(no path separators, no '.' or '..')"
            )
        if name in RESERVED_FILE_NAMES:
            return None, f"error: '{name}' is reserved"
        return os.path.join(scratch_dir, name), None

    def _list_files(**_kwargs) -> str:
        if scratch_dir is None:
            return "files unavailable (no scratchpad directory)"
        try:
            entries = sorted(os.listdir(scratch_dir))
        except FileNotFoundError:
            return "no files yet"
        except OSError as exc:
            return f"error: could not list files: {exc}"
        items = []
        for entry in entries:
            if entry in RESERVED_FILE_NAMES:
                continue
            full = os.path.join(scratch_dir, entry)
            if not os.path.isfile(full):
                continue
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0
            items.append({"name": entry, "size": size})
        if not items:
            return "no files yet"
        return json.dumps(items, indent=2)

    def _read_file(name: str = "", **_kwargs) -> str:
        path, err = _safe_file_path(name)
        if err:
            return err
        if not os.path.isfile(path):
            return f"error: '{name}' not found"
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = f.read(FILE_MAX_BYTES + 1)
        except OSError as exc:
            return f"error: could not read '{name}': {exc}"
        except UnicodeDecodeError:
            return f"error: '{name}' is not valid UTF-8 text"
        if len(data) > FILE_MAX_BYTES:
            return (
                data[:FILE_MAX_BYTES]
                + f"\n... (truncated at {FILE_MAX_BYTES} bytes)"
            )
        return data

    def _search_files(query: str = "", **_kwargs) -> str:
        if scratch_dir is None:
            return "files unavailable (no scratchpad directory)"
        if not query or not isinstance(query, str):
            return "error: query is required"
        needle = query.lower()
        try:
            entries = sorted(os.listdir(scratch_dir))
        except FileNotFoundError:
            return "no files yet"
        except OSError as exc:
            return f"error: could not list files: {exc}"

        lines_out: list[str] = []
        total = 0
        for entry in entries:
            if total >= FILE_SEARCH_MAX_TOTAL_MATCHES:
                lines_out.append("... (match cap reached)")
                break
            if entry in RESERVED_FILE_NAMES:
                continue
            full = os.path.join(scratch_dir, entry)
            if not os.path.isfile(full):
                continue
            try:
                with open(full, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read(FILE_MAX_BYTES)
            except OSError:
                continue
            hits: list[str] = []
            for i, line in enumerate(text.splitlines(), start=1):
                if needle in line.lower():
                    hits.append(f"  {i}: {line.strip()[:200]}")
                    if len(hits) >= FILE_SEARCH_MAX_MATCHES_PER_FILE:
                        hits.append("  ... (more matches in this file)")
                        break
            if hits:
                lines_out.append(f"[{entry}]")
                lines_out.extend(hits)
                total += len(hits)
        if not lines_out:
            return f"no matches for {query!r}"
        return "\n".join(lines_out)

    def _write_file(name: str = "", content: str = "", **_kwargs) -> str:
        path, err = _safe_file_path(name)
        if err:
            return err
        if not isinstance(content, str):
            return "error: content must be a string"
        encoded = content.encode("utf-8")
        if len(encoded) > FILE_MAX_BYTES:
            return (
                f"error: content is {len(encoded)} bytes, max is "
                f"{FILE_MAX_BYTES}"
            )
        # Enforce max file count only when creating a new file.
        if not os.path.exists(path):
            try:
                existing = [
                    e for e in os.listdir(scratch_dir)
                    if e not in RESERVED_FILE_NAMES
                    and os.path.isfile(os.path.join(scratch_dir, e))
                ]
            except OSError:
                existing = []
            if len(existing) >= FILE_MAX_COUNT:
                return (
                    f"error: file limit reached ({FILE_MAX_COUNT}); "
                    "delete or overwrite an existing file"
                )
        try:
            os.makedirs(scratch_dir, exist_ok=True)
            with open(path, "wb") as f:
                f.write(encoded)
        except OSError as exc:
            return f"error: could not write '{name}': {exc}"
        return f"wrote {name} ({len(encoded)} bytes)"

    # ── Control ──────────────────────────────────────────────────

    def _submit(code: str = "", name: str = "", motivation: str = "",
                **_kwargs) -> str:
        # One-shot nag: if the LLM hasn't written any scratchpad note
        # this round, ask once before we ship. On the second submit we
        # accept even without a note — a determined LLM isn't blocked,
        # we just won't have fresh notes for the next round.
        if (not state_holder.get("wrote_this_round")
                and state_holder.get("submit_nag_count", 0) == 0):
            state_holder["submit_nag_count"] = 1
            return (
                "error: please write at least one note via "
                "`write_scratchpad` before submitting — pass one of "
                "`hypothesis`, `dead_end` + `reason`, or "
                "`observation`. Future rounds need your observations "
                "from this one. This is a one-time reminder; if you "
                "call `submit` again without writing, the submission "
                "will go through."
            )
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
        "size_to_flops": _size_to_flops,
        "list_frontier": _list_frontier,
        "get_frontier_member": _get_frontier_member,
        "trace_architecture": _trace_architecture,
        "check_output_shape": _check_output_shape,
        "validate_code": _validate_code,
        "read_scratchpad": _read_scratchpad,
        "write_scratchpad": _write_scratchpad,
        "list_files": _list_files,
        "read_file": _read_file,
        "write_file": _write_file,
        "search_files": _search_files,
        "submit": _submit,
        "time_remaining": _time_remaining,
    }

    def _wrap(name, fn):
        def wrapped(**kwargs):
            _call_counts[name] = _call_counts.get(name, 0) + 1
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
        # persist scratchpad notes after the loop. Expose the per-round
        # counter the same way so the agent can log a usage summary.
        if name == "submit":
            wrapped._state_holder = state_holder  # type: ignore[attr-defined]
            wrapped._call_counts = _call_counts   # type: ignore[attr-defined]
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
