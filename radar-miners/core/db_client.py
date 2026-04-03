"""Validator DB query client — REST endpoints, no auth needed."""

import json
import sys
import urllib.request
import urllib.error

TIMEOUT = 10


def _get(base_url: str, path: str, params: dict | None = None) -> dict | list:
    """GET request with graceful fallback."""
    url = f"{base_url.rstrip('/')}{path}"
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{qs}"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        print(f"[db] GET {path} failed: {exc}", file=sys.stderr)
        return {}


def _post(base_url: str, path: str, body: dict) -> dict | list:
    """POST request with graceful fallback."""
    url = f"{base_url.rstrip('/')}{path}"
    try:
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        print(f"[db] POST {path} failed: {exc}", file=sys.stderr)
        return {}


def recent_experiments(base_url: str, n: int = 15) -> dict | list:
    return _get(base_url, "/experiments/recent", {"n": n})


def pareto_front(base_url: str) -> dict | list:
    return _get(base_url, "/experiments/pareto")


def recent_failures(base_url: str, n: int = 5) -> dict | list:
    return _get(base_url, "/experiments/failures", {"n": n})


def family_summaries(base_url: str) -> dict | list:
    return _get(base_url, "/experiments/families")


def component_stats(base_url: str) -> dict | list:
    return _get(base_url, "/provenance/component_stats")


def dead_ends(base_url: str) -> dict | list:
    return _get(base_url, "/provenance/dead_ends")


def similar_experiments(base_url: str, exp_id: str) -> dict | list:
    return _get(base_url, f"/provenance/{exp_id}/similar")


def search_experiments(base_url: str, query: str) -> dict | list:
    return _post(base_url, "/experiments/search", {"query": query})
