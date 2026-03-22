"""
tests/conftest.py

Root conftest: two responsibilities

1. sys.path setup — add every service root so tests can import from each
   package (gateway, scheduler, node_agent) without editable installs.

2. CI metrics plugin — collects per-marker pass/fail counts + test durations
   and writes  test-results/ci_report.json  at the end of the session.
   The CI workflow uploads this as an artifact.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# 1. sys.path setup
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))

# Nested packages (service/service/__init__.py) — add the service directory so
# that `import scheduler`, `import node_agent`, `import mrm` resolve correctly.
for _service in ("scheduler", "node_agent", "model_runtime_manager"):
    _path = os.path.join(_REPO_ROOT, _service)
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

# Flat package (gateway/__init__.py lives directly in the service directory) —
# add the repo root so that `from gateway.services.X import …` resolves.
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# 2. CI metrics plugin
# ---------------------------------------------------------------------------

_BEHAVIORAL_MARKERS = {
    "invariant", "resilience", "streaming", "concurrency", "slow", "chaos",
}


class _CIMetricsPlugin:
    """
    Lightweight pytest plugin that aggregates test results into a structured
    JSON report.

    The report captures:
    - Summary counts (total / passed / failed / skipped / xfailed)
    - Per-marker breakdown
    - High-level behavioral status (idempotency, failover, streaming, …)
    - p95 latency proxy derived from test execution durations
    - Overall error rate

    Written to  test-results/ci_report.json  at session end.
    """

    def __init__(self) -> None:
        # nodeid → set of marker names
        self._item_markers: dict[str, set[str]] = {}
        # Collected test results
        self._records: list[dict] = []

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        """Cache marker names for every collected item."""
        for item in session.items:
            self._item_markers[item.nodeid] = {
                m.name for m in item.iter_markers()
            }

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Record the outcome of every test's call phase."""
        if report.when != "call":
            return
        markers = self._item_markers.get(report.nodeid, set())
        if report.passed:
            status = "passed"
        elif report.skipped:
            # xfail shows as skipped with wasxfail attribute
            status = "xfailed" if hasattr(report, "wasxfail") else "skipped"
        else:
            status = "failed"

        self._records.append({
            "nodeid": report.nodeid,
            "status": status,
            "duration_s": getattr(report, "duration", 0.0),
            "markers": list(markers & _BEHAVIORAL_MARKERS),
        })

    def pytest_sessionfinish(
        self, session: pytest.Session, exitstatus: int
    ) -> None:
        """Write the CI report after all tests have run."""
        if not self._records:
            return

        total = len(self._records)
        passed = sum(1 for r in self._records if r["status"] == "passed")
        failed = sum(1 for r in self._records if r["status"] == "failed")
        skipped = sum(1 for r in self._records if r["status"] == "skipped")
        xfailed = sum(1 for r in self._records if r["status"] == "xfailed")

        # Per-marker breakdown
        by_marker: dict[str, dict] = {}
        for rec in self._records:
            for marker in rec["markers"]:
                if marker not in by_marker:
                    by_marker[marker] = {"passed": 0, "failed": 0}
                if rec["status"] == "passed":
                    by_marker[marker]["passed"] += 1
                elif rec["status"] == "failed":
                    by_marker[marker]["failed"] += 1

        # Behavioral status: "ok" if all tests in that category passed
        def _status(marker: str, keywords: list[str]) -> str:
            relevant = [
                r for r in self._records
                if marker in r["markers"]
                or any(kw in r["nodeid"].lower() for kw in keywords)
            ]
            if not relevant:
                return "no_tests"
            failed_tests = [r["nodeid"] for r in relevant if r["status"] == "failed"]
            return "ok" if not failed_tests else f"failed"

        # Latency proxy: p95 of test execution durations (passed tests only)
        durations_ms = sorted(
            r["duration_s"] * 1000
            for r in self._records
            if r["status"] == "passed" and r["duration_s"] > 0
        )
        p95_ms = (
            durations_ms[int(len(durations_ms) * 0.95)]
            if len(durations_ms) >= 2
            else None
        )

        report = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "git_sha": os.environ.get("GITHUB_SHA", "local")[:8],
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "xfailed": xfailed,
            },
            "by_marker": by_marker,
            "status": {
                "idempotency": _status("invariant", ["idempotent", "single_placement"]),
                "failover": _status("resilience", ["failover", "re_place", "dead_node"]),
                "streaming": _status("streaming", ["stream", "sse", "mid_stream"]),
                "concurrency": _status("concurrency", ["concurrent", "deadlock"]),
                "chaos": _status("chaos", ["chaos", "random"]),
            },
            "latency_p95_ms": round(p95_ms, 1) if p95_ms is not None else None,
            "error_rate": round(failed / total, 4) if total > 0 else 0.0,
        }

        out = Path("test-results/ci_report.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))

        # Also print to stdout so it appears in CI logs
        print(f"\n{'='*60}")
        print("CI REPORT")
        print(f"{'='*60}")
        print(json.dumps(report, indent=2))


def pytest_configure(config: pytest.Config) -> None:
    config.pluginmanager.register(_CIMetricsPlugin(), "ci_metrics")
