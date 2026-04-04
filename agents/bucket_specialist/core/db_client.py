"""Validator DB query client — uses GatedClient for all HTTP access."""

import sys


def _get(client, base_url: str, path: str, params: dict | None = None) -> dict | list:
    """GET request with graceful fallback."""
    url = f"{base_url.rstrip('/')}{path}"
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{qs}"
    try:
        return client.get_json(url)
    except Exception as exc:
        print(f"[db] GET {path} failed: {exc}", file=sys.stderr)
        return {}


def _post(client, base_url: str, path: str, body: dict) -> dict | list:
    """POST request with graceful fallback."""
    url = f"{base_url.rstrip('/')}{path}"
    try:
        return client.post_json(url, body)
    except Exception as exc:
        print(f"[db] POST {path} failed: {exc}", file=sys.stderr)
        return {}


def recent_experiments(client, base_url: str, n: int = 15) -> dict | list:
    return _get(client, base_url, "/experiments/recent", {"n": n})


def pareto_front(client, base_url: str) -> dict | list:
    return _get(client, base_url, "/experiments/pareto")


def recent_failures(client, base_url: str, n: int = 5) -> dict | list:
    return _get(client, base_url, "/experiments/failures", {"n": n})


def family_summaries(client, base_url: str) -> dict | list:
    return _get(client, base_url, "/experiments/families")


def component_stats(client, base_url: str) -> dict | list:
    return _get(client, base_url, "/provenance/component_stats")


def dead_ends(client, base_url: str) -> dict | list:
    return _get(client, base_url, "/provenance/dead_ends")


def similar_experiments(client, base_url: str, exp_id: str) -> dict | list:
    return _get(client, base_url, f"/provenance/{exp_id}/similar")


def search_experiments(client, base_url: str, query: str) -> dict | list:
    return _post(client, base_url, "/experiments/search", {"query": query})
