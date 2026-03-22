#!/usr/bin/env python3
"""
scripts/collect_ci_metrics.py

Aggregates per-job JUnit XML results and the pytest CI report into a single
structured JSON artifact.

Output written to: test-results/summary.json

Usage:
  python scripts/collect_ci_metrics.py \
    --unit    test-results/unit.xml \
    --integration test-results/integration.xml \
    --resilience  test-results/resilience.xml \
    --chaos   test-results/chaos.xml \
    --report  test-results/ci_report.json

Exit code: always 0 (this script only aggregates, does not gate CI).
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_junit(path: Path | None) -> dict:
    """Extract summary counts from a JUnit XML file."""
    if path is None or not path.exists():
        return {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0}

    tree = ET.parse(path)
    root = tree.getroot()

    # Aggregate across all suites
    totals = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    suites = root.findall(".//testsuite") or [root]

    for suite in suites:
        tests = int(suite.get("tests", 0))
        failures = int(suite.get("failures", 0))
        errors = int(suite.get("errors", 0))
        skipped = int(suite.get("skipped", 0))

        totals["total"] += tests
        totals["failed"] += failures
        totals["errors"] += errors
        totals["skipped"] += skipped
        totals["passed"] += max(0, tests - failures - errors - skipped)

    return totals


def load_ci_report(path: Path | None) -> dict:
    """Load the pytest-generated CI report JSON if available."""
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def compute_overall_status(jobs: dict[str, dict]) -> str:
    """Overall is 'pass' only if every job has zero failures and errors."""
    for job_data in jobs.values():
        if job_data.get("failed", 0) > 0 or job_data.get("errors", 0) > 0:
            return "fail"
    return "pass"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Aggregate CI test results")
    parser.add_argument("--unit", type=Path, default=None)
    parser.add_argument("--integration", type=Path, default=None)
    parser.add_argument("--resilience", type=Path, default=None)
    parser.add_argument("--chaos", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None, help="pytest ci_report.json")
    parser.add_argument("--output", type=Path, default=Path("test-results/summary.json"))
    args = parser.parse_args(argv[1:])

    jobs = {
        "unit": parse_junit(args.unit),
        "integration": parse_junit(args.integration),
        "resilience": parse_junit(args.resilience),
        "chaos": parse_junit(args.chaos),
    }

    ci_report = load_ci_report(args.report)

    # Pull behavioral status and latency from the pytest CI report if available
    behavioral_status = ci_report.get("status", {})
    latency_p95_ms = ci_report.get("latency_p95_ms")
    error_rate = ci_report.get("error_rate", 0.0)

    summary = {
        "overall": compute_overall_status(jobs),
        "jobs": jobs,
        "status": {
            "idempotency": behavioral_status.get("idempotency", "no_tests"),
            "failover": behavioral_status.get("failover", "no_tests"),
            "streaming": behavioral_status.get("streaming", "no_tests"),
            "concurrency": behavioral_status.get("concurrency", "no_tests"),
            "chaos": behavioral_status.get("chaos", "no_tests"),
        },
        "latency_p95_ms": latency_p95_ms,
        "error_rate": error_rate,
        "git_sha": ci_report.get("git_sha", "unknown"),
        "timestamp": ci_report.get("timestamp", ""),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"Summary written to {args.output}")
    print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
