"""OpenAI-SDK reference miner agent.

Reuses ``core/*`` utilities from the sibling ``agents/autonomous/`` package
via sys.path injection so we don't duplicate FLOPs / validation / history
modules. Importing this package puts ``agents/autonomous/`` on ``sys.path``
so submodules can ``from core.X import Y`` exactly like the autonomous
agent does.
"""

import os
import sys

_AUTONOMOUS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "autonomous")
)
if _AUTONOMOUS_DIR not in sys.path:
    sys.path.insert(0, _AUTONOMOUS_DIR)
