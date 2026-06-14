#!/usr/bin/env bash
#
# Run THIS repo's CLI behavior tests against another Ariadne checkout to verify
# that splitting cli/*.py into ~600-LOC modules preserved behavior.
#
# Why this works: the test *bodies* assert command behavior (persisted DB/fs
# state, ranking, scoping, counts) — they're portable. The only thing coupled
# to file layout is each file's single `from cli.<module> import ...` line.
# This script finds, in the TARGET repo, which cli module actually defines each
# command (e.g. before the split the status commands live in cli.core; after,
# in cli.status), retargets that one import line, and runs the test. So it
# works before, during, and after the split.
#
# The TARGET repo is never modified: tests are staged in a temp dir and run
# read-only via the target's own .venv, with all pytest artifacts in the temp
# dir and a clean cwd.
#
# Usage:  ./run_cli_tests_against.sh <TARGET_REPO>
#         TARGET_REPO: path to another Ariadne checkout (must have a .venv).

set -uo pipefail

SRC_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:?usage: run_cli_tests_against.sh <TARGET_REPO>  (an Ariadne checkout with a .venv)}"
PY="${PYTHON:-$TARGET/.venv/bin/python}"   # PYTHON= overrides (e.g. for a worktree with no .venv)

# test file | module it imports in THIS repo | a probe symbol that locates it in TARGET
TESTS=(
  "test_cli_core.py|cli.core|cmd_search"
  "test_cli_status.py|cli.status|cmd_stats"
  "test_cli_lookup.py|cli.lookup|cmd_symbol"
)

if [ ! -x "$PY" ]; then
  echo "ERROR: no python at $PY (target repo has no .venv?)" >&2
  exit 2
fi

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/cli-tests.XXXXXX")"; trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/bt"

echo "Source tests : $SRC_REPO/tests"
echo "Target repo  : $TARGET"
echo "Python       : $PY"
echo

overall=0
for entry in "${TESTS[@]}"; do
  IFS='|' read -r tf orig probe <<< "$entry"
  src="$SRC_REPO/tests/$tf"
  if [ ! -f "$src" ]; then echo "SKIP  $tf — not found in source repo"; echo; continue; fi

  # Locate the cli.* module that defines the probe symbol in the TARGET repo.
  found="$("$PY" - "$TARGET/cli" "$probe" <<'PYEOF'
import ast, pathlib, sys
cli_dir, probe = pathlib.Path(sys.argv[1]), sys.argv[2]
for f in sorted(cli_dir.glob("*.py")):
    try:
        tree = ast.parse(f.read_text())
    except SyntaxError:
        continue
    names = {n.name for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if probe in names:
        print("cli." + f.stem)
        break
PYEOF
)"
  if [ -z "$found" ]; then
    echo "SKIP  $tf — '$probe' not defined in any $TARGET/cli/*.py (feature absent in target)"
    echo
    continue
  fi

  # Stage the test with only its import line retargeted to where the code lives.
  staged="$STAGE/$tf"
  sed "s/from ${orig//./\\.} import/from ${found} import/" "$src" > "$staged"

  echo "RUN   $tf   ($orig here  ->  $found in target)"
  if ( cd "$STAGE" && PYTHONPATH="$TARGET" "$PY" -m pytest "$staged" \
         -q -p no:cacheprovider --import-mode=prepend --basetemp="$STAGE/bt" ); then
    echo "  -> PASS"
  else
    echo "  -> FAIL — behavior differs, or the split broke something"
    overall=1
  fi
  echo
done

if [ "$overall" -eq 0 ]; then
  echo "ALL APPLICABLE TESTS PASSED — command behavior preserved in the target."
else
  echo "FAILURES above — investigate before trusting the split."
fi
exit "$overall"
