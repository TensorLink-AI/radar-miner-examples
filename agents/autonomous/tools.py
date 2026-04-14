"""Autonomous agent tool definitions — a rich set of tools the LLM can call
to research, generate, validate, and submit model architectures.

The agent decides what to do and when. Tools cover:
  - Research: arxiv search, experiment DB queries
  - Analysis: frontier details, FLOPs estimation
  - Validation: full AST + FLOPs validation
  - State: scratchpad read/write
  - Control: submit (ends loop), time remaining
"""

import json
import sys
import time

from core import validation
from core.flops_estimator import estimate_flops, suggest_resize
from core.history import (
    extract_flops_budget, identify_bucket, load_state, save_state,
    get_history, format_history,
)
from core.output_shape import infer_output_shape, verify_output_shape
from core.prompt_builder import format_frontier
from core.trace import trace_architecture, format_trace

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    # ── Research tools ────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": (
                "Search arxiv for papers relevant to your architecture design. "
                "Returns titles, abstracts, and key findings. Use this to find "
                "state-of-the-art techniques for the task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g. 'efficient time series forecasting transformer')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default 5)",
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
                "Query the experiment database. You can hit any endpoint with "
                "GET or POST. Use this to explore what data is available and "
                "pull whatever information helps your design.\n\n"
                "Known endpoints (but feel free to discover others):\n"
                "  GET  /experiments/recent?n=15      — recent experiment results\n"
                "  GET  /experiments/failures?n=5     — recent failures with reasons\n"
                "  GET  /provenance/component_stats   — which components correlate with success\n"
                "  GET  /provenance/dead_ends         — patterns that consistently fail\n"
                "  GET  /experiments/{id}             — details for a specific experiment\n"
                "  POST /experiments/search           — search with custom filters\n\n"
                "You can also try paths like /experiments/top, /experiments/by_bucket, "
                "/provenance/lineage, etc. — the DB may support more than what's listed."
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
                        "description": "API path, e.g. '/experiments/recent?n=10' or '/provenance/component_stats'",
                    },
                    "body": {
                        "type": "object",
                        "description": "JSON body for POST requests (optional)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    # ── Analysis tools ────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_frontier_details",
            "description": (
                "Get detailed information about the current Pareto frontier — "
                "the models you need to beat. Returns their objectives, code, "
                "and motivations. Study these to find improvement opportunities."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_entries": {
                        "type": "integer",
                        "description": "Max frontier members to return (default 5)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_model_flops",
            "description": (
                "Estimate the FLOPs of a model code snippet WITHOUT full validation. "
                "Use this for quick checks while iterating on your design. "
                "Returns the estimated FLOPs count and whether it fits the budget."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Python code with build_model and build_optimizer",
                    },
                },
                "required": ["code"],
            },
        },
    },
    # ── Validation tools ──────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "trace_architecture",
            "description": (
                "Run a dummy forward pass through build_model and return an "
                "op-by-op trace: each leaf module's name, class, input shape, "
                "output shape, and parameter count. Memory-efficient — only "
                "shape tuples and ints are captured, never tensor data. Use "
                "this to debug where a shape goes wrong inside the model, or "
                "to verify an architecture behaves as you expect before "
                "submission."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Python code with build_model",
                    },
                    "max_rows": {
                        "type": "integer",
                        "description": (
                            "Maximum trace rows to include in the rendered "
                            "output (default 60). Deeper traces are still "
                            "recorded internally but truncated for display."
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
                "Run build_model and a dummy forward pass, then compare the "
                "output tensor's shape against the expected shape parsed from "
                "the task constraints. Catches 'tensor a (X) must match tensor "
                "b (Y)' training failures before submission. Works for any "
                "output rank (2D, 3D, 4D, ...) — the expected shape is "
                "derived from the challenge, never hardcoded."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Python code with build_model",
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
                "Run full validation on your code: syntax check, required functions, "
                "parameter signatures, forbidden imports, and FLOPs bounds. "
                "Returns pass/fail with detailed error messages. "
                "ALWAYS validate before submitting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Python code to validate",
                    },
                },
                "required": ["code"],
            },
        },
    },
    # ── State tools ───────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "read_scratchpad",
            "description": (
                "Load persistent state from scratchpad. Contains history of "
                "your previous submissions across rounds — what you tried, "
                "what worked, what didn't. Check this first each round."
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
                "Save data to persistent scratchpad for future rounds. "
                "Use this to record what you learned, what worked, and "
                "what to try next time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "notes": {
                        "type": "string",
                        "description": "Free-form notes to save for future rounds",
                    },
                },
                "required": [],
            },
        },
    },
    # ── Control tools ─────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": (
                "Submit your final model code. This ends the loop. "
                "The code MUST have passed validate_code first. "
                "Include a descriptive name and motivation explaining "
                "your design choices."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The validated Python code to submit",
                    },
                    "name": {
                        "type": "string",
                        "description": "Short descriptive name for this submission",
                    },
                    "motivation": {
                        "type": "string",
                        "description": "Why you designed the model this way — what you expect to improve",
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
                "Check how many seconds remain in the time budget. "
                "Plan your work accordingly — if time is low, submit "
                "what you have rather than starting new research."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Handler factory
# ---------------------------------------------------------------------------

class SubmitSignal(Exception):
    """Raised when the agent calls submit — carries the result out of the loop."""
    def __init__(self, code: str, name: str, motivation: str):
        self.code = code
        self.name = name
        self.motivation = motivation


def build_handlers(client, challenge: dict, scratch_dir, deadline: float) -> dict:
    """Build tool handler callables that close over agent state.

    Each handler returns a string for the LLM to see as tool output.
    The ``submit`` handler raises ``SubmitSignal`` to break the loop.
    """
    db_url = challenge.get("db_url", "")
    desearch_url = challenge.get("desearch_url", "")
    flops_min, flops_max = extract_flops_budget(challenge)
    target_flops = int(flops_max * 0.6) if flops_max else 0

    # Mutable state holder for scratchpad notes
    state_holder = {"state": load_state(scratch_dir) if scratch_dir else {}}

    def _fmt(data) -> str:
        try:
            return json.dumps(data, default=str, indent=2)
        except (TypeError, ValueError):
            return str(data)

    # ── Research handlers ─────────────────────────────────────────

    def search_papers(query: str, max_results: int = 5) -> str:
        if not desearch_url:
            return "Paper search unavailable (no desearch_url)."
        try:
            resp = client.post_json(
                f"{desearch_url}/search",
                {"query": query, "max_results": max_results},
            )
            results = resp.get("results", [])
            if not results:
                return "No papers found."
            lines = []
            for r in results[:max_results]:
                lines.append(f"**{r.get('title', 'untitled')}**")
                abstract = r.get("abstract", "")
                if abstract:
                    lines.append(abstract[:500])
                lines.append("")
            return "\n".join(lines)
        except Exception as exc:
            return f"Paper search failed: {exc}"

    def query_db(path: str, method: str = "GET", body: dict | None = None) -> str:
        if not db_url:
            return "Experiment DB unavailable (no db_url in challenge)."
        # Build the full URL: db_url base + path
        base = db_url.rstrip("/")
        # Ensure path starts with /
        if not path.startswith("/"):
            path = "/" + path
        url = f"{base}{path}"
        try:
            if method.upper() == "POST":
                result = client.post_json(url, body or {})
            else:
                result = client.get_json(url)
            if isinstance(result, dict) and "error" in result:
                return f"DB returned error: {result['error']}"
            formatted = _fmt(result)
            # Truncate very large responses to avoid flooding context
            if len(formatted) > 8000:
                formatted = formatted[:8000] + "\n... (truncated, try a more specific query)"
            return formatted
        except Exception as exc:
            print(f"[tools] query_db {method} {path} failed: {exc}",
                  file=sys.stderr)
            return f"DB query failed: {exc}"

    # ── Analysis handlers ─────────────────────────────────────────

    def get_frontier_details(max_entries: int = 5) -> str:
        frontier = challenge.get("feasible_frontier", [])
        if not frontier:
            frontier = challenge.get("pareto_frontier", [])
        if not frontier:
            return "No frontier exists yet — you are bootstrapping the first model."
        return format_frontier(frontier, max_entries=max_entries)

    def estimate_model_flops(code: str) -> str:
        estimated, err = estimate_flops(code, challenge)
        if err:
            return f"FLOPs estimation failed: {err}"
        gate_min = int(flops_min * 0.9)
        gate_max = int(flops_max * 1.1)
        status = "WITHIN BUDGET"
        hint = ""
        if estimated < gate_min:
            status = f"TOO LOW (below gate minimum {gate_min:,})"
            hint = suggest_resize(estimated, gate_min, gate_max, target_flops)
        elif estimated > gate_max:
            status = f"TOO HIGH (above gate maximum {gate_max:,})"
            hint = suggest_resize(estimated, gate_min, gate_max, target_flops)
        result = (
            f"Estimated FLOPs: {estimated:,}\n"
            f"Target: {target_flops:,} (60% of max)\n"
            f"Budget: [{flops_min:,}, {flops_max:,}]\n"
            f"Hard gate: [{gate_min:,}, {gate_max:,}]\n"
            f"Status: {status}"
        )
        if hint:
            result += f"\n{hint}"
        return result

    # ── Validation handler ────────────────────────────────────────

    def trace_architecture_handler(code: str, max_rows: int = 60) -> str:
        entries, err = trace_architecture(code, challenge)
        if err:
            return f"Trace failed: {err}"
        if not entries:
            return (
                "Trace produced no entries — the model has no leaf modules "
                "with parameters, or the forward pass produced no tensor "
                "outputs through hooked modules."
            )
        return format_trace(entries, max_rows=max_rows)

    def check_output_shape_handler(code: str) -> str:
        task = challenge.get("task", {}) or {}
        tp = task.get("task_params", {}) or {}
        constraints = task.get("constraints", []) or []
        expected = infer_output_shape(tp, constraints)
        if expected is None:
            return (
                "No parseable output-shape constraint in task.constraints; "
                "skipping shape check. (This is not an error — the task "
                "simply doesn't declare an expected output shape.)"
            )
        sink: list = []
        _, err = estimate_flops(code, challenge, sink)
        if err:
            return f"Could not run forward pass: {err}"
        if not sink:
            return (
                "Forward pass completed but no tensor output was captured "
                "(possibly via a fallback path). Add a concrete forward() "
                "that returns a torch.Tensor."
            )
        shape_err = verify_output_shape(sink[0], expected)
        if shape_err:
            return f"FAILED: {shape_err}"
        return (
            f"PASSED: Output shape {tuple(sink[0])} matches expected "
            f"(B, {', '.join(str(e) if e >= 0 else '?' for e in expected)})."
        )

    def validate_code_handler(code: str) -> str:
        ok, errors = validation.validate_code(code, challenge)
        if ok:
            return "PASSED: Code is valid and FLOPs are within budget."
        return "FAILED:\n" + "\n".join(f"- {e}" for e in errors)

    # ── State handlers ────────────────────────────────────────────

    def read_scratchpad() -> str:
        state = state_holder["state"]
        history_entries = get_history(state)
        notes = state.get("agent_notes", "")
        parts = []
        if notes:
            parts.append(f"## Saved Notes\n{notes}")
        if history_entries:
            parts.append(f"## Submission History\n{format_history(history_entries, max_entries=10)}")
        if not parts:
            return "Scratchpad is empty — this is your first round."
        return "\n\n".join(parts)

    def write_scratchpad(notes: str = "") -> str:
        state = state_holder["state"]
        if notes:
            state["agent_notes"] = notes
        state_holder["state"] = state
        return "Scratchpad updated."

    # ── Control handlers ──────────────────────────────────────────

    def submit_handler(code: str, name: str, motivation: str) -> str:
        # Validate one last time before accepting
        ok, errors = validation.validate_code(code, challenge)
        if not ok:
            return (
                "SUBMIT REJECTED — code failed validation:\n"
                + "\n".join(f"- {e}" for e in errors)
                + "\n\nFix the errors and try again."
            )
        raise SubmitSignal(code, name, motivation)

    def time_remaining_handler() -> str:
        remaining = max(0, deadline - time.time())
        if remaining > 60:
            return f"{remaining:.0f} seconds remaining ({remaining / 60:.1f} minutes)."
        elif remaining > 0:
            return f"{remaining:.0f} seconds remaining — wrap up soon!"
        return "Time is up! Submit now or the round will end without a submission."

    # Expose state_holder so the agent loop can persist state after the loop
    submit_handler._state_holder = state_holder

    return {
        "search_papers": search_papers,
        "query_db": query_db,
        "get_frontier_details": get_frontier_details,
        "estimate_model_flops": estimate_model_flops,
        "trace_architecture": trace_architecture_handler,
        "check_output_shape": check_output_shape_handler,
        "validate_code": validate_code_handler,
        "read_scratchpad": read_scratchpad,
        "write_scratchpad": write_scratchpad,
        "submit": submit_handler,
        "time_remaining": time_remaining_handler,
    }
