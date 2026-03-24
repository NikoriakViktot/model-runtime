"""
shared/runtime_types.py

Canonical type definitions shared across all services.
Import from here — never duplicate these literals.
"""

from __future__ import annotations

from typing import Literal

# Runtime backend: GPU (vLLM) or CPU (llama.cpp / GGUF)
RuntimeType = Literal["gpu", "cpu"]

# Model size hint — used for auto-routing decisions
ModelSize = Literal["small", "medium", "large"]

# Latency class — workload sensitivity signal
LatencyClass = Literal["fast", "slow"]

# Client-facing override: explicit preference or let the gateway decide
RuntimePreference = Literal["gpu", "cpu", "auto"]

# Routing decision result (internal gateway type)
RoutingDecision = Literal["gpu", "cpu"]
