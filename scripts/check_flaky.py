#!/usr/bin/env python3
"""
scripts/check_flaky.py

Detects flaky tests by comparing N JUnit XML result files from repeated
test runs.  A test is "flaky" if it passed in at least one run and failed
in at least one other run.

Exit code:
  0 — no flaky tests detected
  1 — at least one flaky test detected (CI should fail)

Usage:
  python scripts/check_flaky.py results/run1.xml results/run2.xml results/run3.xml
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_results(path: Path) -> dict[str, str]:
    """
    Parse a JUnit XML file and return {test_nodeid: outcome}.
    Outcome is "passed", "failed", or "error".
    """
    results: dict[str, str] = {}
    tree = ET.parse(path)
    root = tree.getroot()

    # Handle both <testsuites> wrapper and bare <testsuite> root
    suites = root.findall(".//testsuite") or [root]

    for suite in suites:
        for case in suite.findall("testcase"):
            classname = case.get("classname", "")
            name = case.get("name", "")
            nodeid = f"{classname}::{name}" if classname else name

            if case.find("failure") is not None or case.find("error") is not None:
                results[nodeid] = "failed"
            elif case.find("skipped") is not None:
                results[nodeid] = "skipped"
            else:
                results[nodeid] = "passed"

    return results


def detect_flaky(run_results: list[dict[str, str]]) -> list[str]:
    """
    Return a list of test node IDs that have inconsistent outcomes across runs.
    A test is flaky if it both passed in at least one run and failed in another.
    """
    all_nodeids: set[str] = set()
    for results in run_results:
        all_nodeids.update(results.keys())

    flaky: list[str] = []
    for nodeid in sorted(all_nodeids):
        outcomes = {results.get(nodeid) for results in run_results if nodeid in results}
        # Flaky = has both passing and failing outcomes
        if "passed" in outcomes and "failed" in outcomes:
            flaky.append(nodeid)

    return flaky


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: check_flaky.py <run1.xml> [run2.xml ...]", file=sys.stderr)
        return 2

    xml_paths = [Path(p) for p in argv[1:]]
    for p in xml_paths:
        if not p.exists():
            print(f"ERROR: result file not found: {p}", file=sys.stderr)
            return 2

    run_results = [parse_results(p) for p in xml_paths]

    total_tests = len({nid for r in run_results for nid in r})
    print(f"Checked {len(xml_paths)} runs, {total_tests} unique tests")

    flaky = detect_flaky(run_results)

    if not flaky:
        print("✓ No flaky tests detected")
        return 0

    print(f"\n✗ {len(flaky)} FLAKY test(s) detected:\n")
    for nodeid in flaky:
        outcomes_by_run = [r.get(nodeid, "missing") for r in run_results]
        print(f"  FLAKY  {nodeid}")
        print(f"         outcomes: {outcomes_by_run}")

    print(
        f"\nFlaky tests indicate non-deterministic behavior. "
        f"Investigate before merging."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
