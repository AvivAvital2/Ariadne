#!/usr/bin/env bash
#
# Build the eval store ONCE: an httpx mini-environment spool + the three
# archetype consumer fixtures, onboarded into evals/store/.
#
# Costs real money (small): batched embeddings + doc generation for httpx
# at a pinned tag plus three tiny fixture repos — every paid step shows a
# cost preview first. Needs: git, scip-python on PATH, OPENAI_API_KEY (or
# an OPENAI_BASE_URL-compatible endpoint). After this, evals/run_battery.py
# runs offline forever (query vectors cache on first run).
#
#   evals/build_eval_store.sh                    # build (idempotent)
#   HTTPX_TAG=v0.28.1 evals/build_eval_store.sh  # override the pinned tag
set -euo pipefail

EVALS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$EVALS_DIR")"
STORE="$EVALS_DIR/store"
HTTPX_TAG="${HTTPX_TAG:-v0.28.1}"

mkdir -p "$STORE"

# Every ariadne command runs with cwd=STORE, so config/recipe/DB discovery
# stays inside evals/store/ and never touches the repo's own store.
run() { (cd "$STORE" && uv run --project "$REPO_ROOT" ariadne "$@"); }

# Recipe for the mini-environment (written only if absent — edit freely).
if [ ! -f "$STORE/spools.yaml" ]; then
  cat > "$STORE/spools.yaml" <<EOF
name: httpx
runtime: $HTTPX_TAG
version: 1.0.0
languages: [python]
name_aliases:
  - httpx
corpus:
  httpx:
    url: https://github.com/encode/httpx
    tag: $HTTPX_TAG
certify:
  - httpx/docs/
EOF
  echo "wrote $STORE/spools.yaml (httpx @ $HTTPX_TAG)"
fi

# 1. Build + install the httpx mini-environment spool (skipped when
#    already registered).
if ! run spools 2>/dev/null | grep -q 'registered *httpx'; then
  run spools create --yes --batch \
    --dest "$STORE/spool-corpus" --out "$STORE/httpx-eval.zip"
  run spools install "$STORE/httpx-eval.zip"
  HTTPX_TAG="$HTTPX_TAG" python3 - "$STORE/ariadne.yaml" <<'EOF'
import os
import sys

import yaml

path = sys.argv[1]
cfg = yaml.safe_load(open(path)) if os.path.exists(path) else {}
cfg = cfg or {}
cfg.setdefault('spools', {})['httpx'] = {'runtime': os.environ['HTTPX_TAG']}
yaml.safe_dump(cfg, open(path, 'w'), sort_keys=False)
print(f'enabled spool httpx (runtime {os.environ["HTTPX_TAG"]}) in {path}')
EOF
fi

# 2. Onboard the three archetype consumers (tiny; batched).
for consumer in relay:adopter tidereport:peripheral meshsync:integrated; do
  name="${consumer%%:*}"; dir="${consumer##*:}"
  run source add "$name" --path "$EVALS_DIR/fixtures/$dir"
  run discover --source "$name"
  run onboard --source "$name" --approve --batch
done

echo
echo "eval store ready: $STORE"
echo "next: uv run python evals/run_battery.py"
