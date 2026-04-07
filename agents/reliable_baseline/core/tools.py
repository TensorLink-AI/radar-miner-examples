"""Tool definitions for LLM tool-calling — wraps db_client functions.

Exports
-------
TOOLS : list[dict]
    OpenAI function-calling format tool definitions.
build_handlers(client, db_url) -> dict
    Returns {"tool_name": callable(**kwargs) -> str} for use with
    ``llm.chat_with_tools``.
"""

import json


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "recent_experiments",
            "description": (
                "Fetch the most recent experiment results from the validator "
                "database, including metrics like crps, mase, exec_time, and "
                "memory_mb."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of recent experiments to return (default 15)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recent_failures",
            "description": (
                "Fetch recent experiment failures with their error reasons. "
                "Useful for understanding what approaches don't work."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of recent failures to return (default 5)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "component_stats",
            "description": (
                "Fetch component-level success statistics showing which "
                "architectural components (attention, convolution, etc.) "
                "correlate with good performance."
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
            "name": "dead_ends",
            "description": (
                "Fetch patterns that consistently fail or produce poor results. "
                "Use this to avoid repeating known bad approaches."
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

def build_handlers(client, db_url: str) -> dict:
    """Build tool handler callables that close over *client* and *db_url*.

    Each handler calls the corresponding ``db_client`` function and returns
    a JSON-formatted string so the LLM can parse the results.
    """
    from core import db_client

    def _fmt(data) -> str:
        """Format db_client results as a compact JSON string."""
        try:
            return json.dumps(data, default=str, indent=2)
        except (TypeError, ValueError):
            return str(data)

    def recent_experiments(n: int = 15) -> str:
        return _fmt(db_client.recent_experiments(client, db_url, n=n))

    def recent_failures(n: int = 5) -> str:
        return _fmt(db_client.recent_failures(client, db_url, n=n))

    def component_stats() -> str:
        return _fmt(db_client.component_stats(client, db_url))

    def dead_ends() -> str:
        return _fmt(db_client.dead_ends(client, db_url))

    return {
        "recent_experiments": recent_experiments,
        "recent_failures": recent_failures,
        "component_stats": component_stats,
        "dead_ends": dead_ends,
    }
