"""Validator DB query client — uses GatedClient for all HTTP access.

All calls are wrapped in try/except and return empty dicts on failure.
DB context is helpful but never critical — the agent must work without it.
"""

import sys


def _get(client, base_url: str, path: str, params: dict | None = None) -> dict | list:
    """GET request with graceful fallback."""
    url = f"{base_url.rstrip('/')}{path}"
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{qs}"
    try:
        result = client.get_json(url)
        if isinstance(result, dict) and "error" in result:
            print(f"[db] GET {path} returned error: {result['error']}",
                  file=sys.stderr)
            return {}
        print(f"[db] GET {path} -> {type(result).__name__} "
              f"({len(result) if isinstance(result, (list, dict)) else '?'} items)",
              file=sys.stderr)
        return result
    except Exception as exc:
        print(f"[db] GET {path} failed: {exc}", file=sys.stderr)
        return {}


def recent_experiments(client, base_url: str, n: int = 15) -> dict | list:
    """Fetch recent experiment results."""
    return _get(client, base_url, "/experiments/recent", {"n": n})


def recent_failures(client, base_url: str, n: int = 5) -> dict | list:
    """Fetch recent failures with reasons."""
    return _get(client, base_url, "/experiments/failures", {"n": n})


def component_stats(client, base_url: str) -> dict | list:
    """Fetch component success correlations."""
    return _get(client, base_url, "/provenance/component_stats")


def dead_ends(client, base_url: str) -> dict | list:
    """Fetch patterns that consistently fail."""
    return _get(client, base_url, "/provenance/dead_ends")
