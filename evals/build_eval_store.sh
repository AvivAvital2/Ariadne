#!/usr/bin/env bash
#
# Build the eval store ONCE: the Databricks environment + the three
# archetype consumer fixtures, onboarded into evals/store/.
#
# The environment comes from ONE of two paths:
#   1. PACK zip (fast, free): set PACK=/path/to/databricks-dbr17.3-lts*.zip
#      (or drop the zip in the repo root) — installs prebuilt docs and
#      embeddings in minutes, no cloning, no LLM spend.
#   2. Recipe build (reproducible from source): no pack found — builds the
#      pack from the SHIPPED recipe (spool_content/recipes/databricks.yaml,
#      pinned SHAs). Honest cost: scip-java COMPILES Spark (JDK 17 +
#      Maven), several hours, and tens of dollars of batched generation —
#      every paid step shows its cost first.
#
# The consumer fixtures are tiny either way (cents). Needs scip-python on
# PATH; the recipe path also needs scip-java + JDK 17. After this,
# evals/run_battery.py runs offline forever (vectors cache on first run).
set -euo pipefail

EVALS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$EVALS_DIR")"
STORE="$EVALS_DIR/store"
RUNTIME="dbr17.3-lts"

command -v scip-python >/dev/null || {
  echo 'scip-python not found on PATH — the consumer fixtures are Python,' >&2
  echo 'so onboarding needs it: npm install -g @sourcegraph/scip-python' >&2
  exit 1
}

mkdir -p "$STORE"

# Every ariadne command runs with cwd=STORE, so config/recipe/DB discovery
# stays inside evals/store/ and never touches the repo's own store.
run() { (cd "$STORE" && uv run --project "$REPO_ROOT" ariadne "$@"); }

if ! run spools 2>/dev/null | grep -q 'registered *databricks'; then
  # Path 1: a prebuilt pack.
  pack="${PACK:-$(ls "$REPO_ROOT"/databricks-${RUNTIME}*.zip 2>/dev/null | sort | tail -1)}"
  if [ -n "${pack:-}" ] && [ -f "$pack" ]; then
    echo "installing prebuilt pack: $pack"
    run spools install "$pack"
  else
    # Path 2: build from the shipped recipe (pinned SHAs; slow + paid).
    echo 'no pack zip found — building from the shipped recipe.'
    echo 'THIS COMPILES SPARK (scip-java, JDK 17) and spends real money;'
    echo 'every cost is shown before spending. Ctrl+C now to abort.'
    command -v scip-java >/dev/null || {
      echo 'scip-java not found on PATH — install via Coursier:' >&2
      echo '  cs install --contrib scip-java' >&2
      exit 1
    }
    [ -f "$STORE/spools.yaml" ] || cp \
      "$REPO_ROOT/spool_content/recipes/databricks.yaml" "$STORE/spools.yaml"
    run spools create --dest "$STORE/spool-corpus" \
      --out "$STORE/databricks-eval.zip" --batch
    run spools install "$STORE/databricks-eval.zip"
  fi
  RUNTIME="$RUNTIME" python3 - "$STORE/ariadne.yaml" <<'EOF'
import os
import sys

import yaml

path = sys.argv[1]
cfg = (yaml.safe_load(open(path)) if os.path.exists(path) else {}) or {}
cfg.setdefault('spools', {})['databricks'] = {
    'runtime': os.environ['RUNTIME']}
yaml.safe_dump(cfg, open(path, 'w'), sort_keys=False)
print(f'enabled spool databricks (runtime {os.environ["RUNTIME"]}) in {path}')
EOF
fi

# Onboard the three archetype consumers (tiny; batched).
for consumer in featureflow:adopter harvest:peripheral lakerun:integrated; do
  name="${consumer%%:*}"; dir="${consumer##*:}"
  run source add "$name" --path "$EVALS_DIR/fixtures/$dir"
  run discover --source "$name"
  run onboard --source "$name" --approve --batch
done

echo
echo "eval store ready: $STORE"
echo "next: uv run python evals/run_battery.py"
