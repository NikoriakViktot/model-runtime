"""
tests/conftest.py

Root conftest: add every service root to sys.path so tests can import
directly from each package (gateway, scheduler, node_agent) without
requiring editable installs.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))

for _service in ("gateway", "scheduler", "node_agent", "model_runtime_manager"):
    _path = os.path.join(_REPO_ROOT, _service)
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)
