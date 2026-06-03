#!/usr/bin/env python3
"""Run pytest tests and report based on expected outcome.

Usage:
    uv run python check_tests.py green tests/test_foo.py tests/test_bar.py -v
    uv run python check_tests.py red tests/test_foo.py -k "test_broken"

First arg is the expectation (green=all pass, red=all fail).
Remaining args are forwarded to pytest.
"""

import json
import subprocess
import sys
import tempfile
import os

RED = "\033[91m"
GREEN = "\033[92m"
BOLD = "\033[1m"
RESET = "\033[0m"

def run_pytest(pytest_args: list[str], report_path: str) -> int:
    cmd = [
        sys.executable, "-m", "pytest",
        f"--tb=long",
        f"--json-report",
        f"--json-report-file={report_path}",
        *pytest_args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def load_report(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def report_green(report: dict) -> bool:
    """Green mode: we expect all tests to pass. Report failures.

    ``skipped`` is treated as non-failure: pytest.skip() is a deliberate
    "don't run me here" signal, not a behavioral failure of the code
    under test.
    """
    tests = report.get("tests", [])

    if not tests:
        print(f"{RED}{BOLD}✗ 0 tests collected — aborting.{RESET}")
        return False

    failures = [t for t in tests if t["outcome"] not in ("passed", "skipped")]

    if not failures:
        total = len(tests)
        print(f"{GREEN}{BOLD}✓ All {total} tests passed as expected.{RESET}")
        return True

    print(f"{RED}{BOLD}✗ {len(failures)} test(s) failed unexpectedly:{RESET}\n")
    for t in failures:
        nodeid = t["nodeid"]
        print(f"  {RED}FAIL{RESET} {nodeid}")
        longrepr = (
            t.get("call", {}).get("longrepr", "")
            or t.get("setup", {}).get("longrepr", "")
        )
        if longrepr:
            indented = "\n".join(f"    {line}" for line in longrepr.splitlines())
            print(f"{indented}\n")
    return False


def report_red(report: dict) -> bool:
    """Red mode: we expect all tests to fail. Report passes."""
    tests = report.get("tests", [])

    if not tests:
        print(f"{RED}{BOLD}✗ 0 tests collected — aborting.{RESET}")
        return False

    passes = [t for t in tests if t["outcome"] == "passed"]

    if not passes:
        total = len(tests)
        print(f"{GREEN}{BOLD}✓ All {total} tests failed as expected.{RESET}")
        return True

    print(f"{RED}{BOLD}✗ {len(passes)} test(s) passed unexpectedly:{RESET}\n")
    for t in passes:
        nodeid = t["nodeid"]
        print(f"  {RED}PASSED (unexpected){RESET} {nodeid}")
    print()
    return False


def main() -> int:
    os.environ["UV_CACHE_DIR"] = os.path.join(os.getcwd(), ".uv-cache")
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <green|red> <pytest args...>", file=sys.stderr)
        return 2

    mode = sys.argv[1].lower()
    if mode not in ("green", "red"):
        print(f"First argument must be 'green' or 'red', got '{mode}'", file=sys.stderr)
        return 2

    pytest_args = sys.argv[2:]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        report_path = tmp.name

    returncode, stdout, stderr = run_pytest(pytest_args, report_path)

    try:
        report = load_report(report_path)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"{RED}Failed to read pytest JSON report.{RESET}", file=sys.stderr)
        print(stdout)
        print(stderr, file=sys.stderr)
        return 1

    if mode == "green":
        ok = report_green(report)
    else:
        ok = report_red(report)

    if not ok and not report.get("tests"):
        # 0 tests collected — surface the actual pytest output so the
        # caller can see why (collection errors, missing conftest, etc.).
        if stdout:
            print(stdout)
        if stderr:
            print(stderr, file=sys.stderr)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
