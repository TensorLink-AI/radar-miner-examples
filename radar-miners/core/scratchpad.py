"""R2 scratchpad — load/save persistent state via presigned URLs."""

import io
import json
import os
import sys
import tarfile
import tempfile
import urllib.request
import urllib.error

MAX_SIZE = 10 * 1024 * 1024  # 10 MB


def load(challenge: dict) -> dict:
    """Download and extract scratchpad state from R2. Returns dict or {}."""
    url = challenge.get("scratchpad", {}).get("download_url", "")
    if not url:
        return {}
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        buf = io.BytesIO(data)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            for member in tar.getmembers():
                if member.name == "state.json":
                    f = tar.extractfile(member)
                    if f:
                        return json.loads(f.read().decode())
        return {}
    except Exception as exc:
        print(f"[scratchpad] load failed: {exc}", file=sys.stderr)
        return {}


def save(challenge: dict, state: dict) -> bool:
    """Package state as tar.gz and upload to R2. Returns success bool."""
    url = challenge.get("scratchpad", {}).get("upload_url", "")
    if not url:
        return False
    try:
        state_bytes = json.dumps(state, indent=2).encode()
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            info = tarfile.TarInfo(name="state.json")
            info.size = len(state_bytes)
            tar.addfile(info, io.BytesIO(state_bytes))

        payload = buf.getvalue()
        if len(payload) > MAX_SIZE:
            print(f"[scratchpad] state too large: {len(payload)} bytes",
                  file=sys.stderr)
            return False

        req = urllib.request.Request(url, data=payload, method="PUT")
        req.add_header("Content-Type", "application/gzip")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status in (200, 201, 204)
    except Exception as exc:
        print(f"[scratchpad] save failed: {exc}", file=sys.stderr)
        return False
