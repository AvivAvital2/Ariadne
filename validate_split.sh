#!/usr/bin/env bash
#
# Comprehensive apples-to-apples validation of a behavior-preserving refactor
# (e.g. splitting cli/*.py into ~600-LOC modules). Three checks:
#
#   1. FULL suite @ OLD_REF (old tests on old code)  — baseline.
#   2. FULL suite @ NEW_REF (new tests on new code)  — the split's code.
#   3. REGRESSIONS = tests that FAIL on NEW but PASS on OLD. For a
#      behavior-preserving refactor this set must be empty. (New tests and
#      pre-existing/environment failures are excluded by construction.)
#   4. HONESTY: the genuinely-NEW test files run against OLD code — they must
#      pass on the pre-split code, proving they characterize real behavior
#      rather than mirror the new implementation.
#
# Worktrees are checked out read-only under $TMPDIR and removed afterwards.
#
# Usage:  ./validate_split.sh [OLD_REF] [NEW_REF]      (defaults: main HEAD)
#
# Note: a failure that appears on BOTH refs (or only in the regression diff due
# to a flaky/environment test such as a uv-subprocess sandbox test) is NOT a
# split regression. On a clean machine the regression set should be empty.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLD_REF="${1:-main}"
NEW_REF="${2:-HEAD}"
PY="$REPO/.venv/bin/python"
RUN="$REPO/run_cli_tests_against.sh"
TB="${TMPDIR:-/tmp}"

[ -x "$PY" ]  || { echo "ERROR: no venv python at $PY" >&2; exit 2; }
[ -x "$RUN" ] || { echo "ERROR: missing $RUN" >&2; exit 2; }

mkwt () {  # <ref> -> echoes worktree path
  # NOTE: separate `local` statements — `local a=$1 b=...$a...` expands $a
  # before it is assigned, which is unbound under `set -u`.
  local ref="$1"
  local wt="$TB/vs-$(echo "$ref" | tr -c 'A-Za-z0-9' '_')"
  git -C "$REPO" worktree remove --force "$wt" >/dev/null 2>&1 || true
  git -C "$REPO" worktree add --quiet --detach "$wt" "$ref" >/dev/null
  echo "$wt"
}
rmwt () { git -C "$REPO" worktree remove --force "$1" >/dev/null 2>&1 || true; }

run_full () {  # <worktree> <failures-out> -> prints the pytest summary line
  local wt="$1" ff="$2"
  ( cd "$wt" && PYTHONPATH="$wt" "$PY" -m pytest tests/ -q -p no:cacheprovider \
      --import-mode=prepend --basetemp="$wt/.bt" ) > "$wt/.log" 2>&1 || true
  grep -E '^(FAILED|ERROR)' "$wt/.log" | sed -E 's/ -.*//; s/^(FAILED|ERROR) //' \
    | sort -u > "$ff"
  grep -E '[0-9]+ (passed|failed|error)' "$wt/.log" | tail -1
}

OWT="$(mkwt "$OLD_REF")"
NWT="$(mkwt "$NEW_REF")"
# Fail loudly if a worktree wasn't created — never silently fall back to cwd
# (that would run both checks against the current tree and give a false pass).
for w in "$OWT" "$NWT"; do
  [ -n "$w" ] && [ -d "$w/cli" ] || { echo "ERROR: worktree not created ('$w')" >&2; exit 3; }
done
OF="$TB/vs-old.fail"; NF="$TB/vs-new.fail"

echo "===== FULL SUITE @ $OLD_REF (old code) ====="; run_full "$OWT" "$OF"
echo "===== FULL SUITE @ $NEW_REF (new code) ====="; run_full "$NWT" "$NF"
echo
echo "===== REGRESSIONS: fail on $NEW_REF but pass on $OLD_REF ====="
REG="$(comm -13 "$OF" "$NF")"
if [ -z "$REG" ]; then
  echo "  none — no test that the split could break started failing"
else
  echo "$REG" | sed 's/^/  /'
  echo "  (verify each isn't a flaky/environment test by running it on both refs)"
fi
echo
echo "===== HONESTY CHECK: the NEW test files vs OLD code ====="
PYTHON="$PY" "$RUN" "$OWT" | grep -E "RUN|PASS|FAIL|ALL APPLICABLE|FAILURES"

rmwt "$OWT"; rmwt "$NWT"
echo
if [ -z "$REG" ]; then
  echo "VERDICT: split preserved behavior across the full suite."
  exit 0
fi
echo "VERDICT: regression candidates above — confirm they are real (not flaky env tests)."
exit 1
