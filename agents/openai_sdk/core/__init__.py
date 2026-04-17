"""Shared helpers for the openai_sdk agent (FLOPs, validation, history, etc.).

Copied verbatim from ``agents/autonomous/core/`` so this agent can be
deployed as a self-contained package — the miner's harness copies this
directory to ``/workspace/agent/`` and loads ``agent.py`` without any
package context, so there is no way to reach a sibling directory at
runtime.
"""
