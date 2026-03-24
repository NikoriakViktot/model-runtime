"""
cpu_runtime/state.py

Shared mutable process-level state accessed by multiple modules.
Using a dedicated module avoids circular imports between app.py and routes.
"""
from __future__ import annotations

# Set to True by the SIGTERM handler; routes check this to return 503.
shutting_down: bool = False
