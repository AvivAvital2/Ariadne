#!/usr/bin/env bash
# Container-journey SUITE — drives all three variants of
# cypress/e2e/container-journey.cy.js:
#   • openai    — gpt generation + OpenAI embeddings    (one key:  OPENAI_API_KEY)
#   • anthropic — Claude generation + OpenAI embeddings  (both keys)
#   • qwen      — local HTTP fixtures (no keys or paid inference)
#
# Usage (from web/e2e/):
#   ./run-container-suite.sh              # SEQUENTIAL: openai, anthropic, qwen
#   ./run-container-suite.sh --parallel   # SIMULTANEOUS (isolated project/ports/image)
#
# Each variant runs as its own `cypress run` with a distinct compose project,
# host ports, and image tag (see the spec's VARIANTS), so --parallel is
# collision-free. Runs all variants even if one fails, then reports; exits
# non-zero if any failed.
#
# Requires: Node 20+ (`nvm use 22` first), a reachable Docker daemon, and the
# key(s) in the repo .env or the shell — OPENAI_API_KEY always, ANTHROPIC_API_KEY
# for the anthropic variant. Each variant runs a REAL onboard + ask, so it costs
# a little (see the spec header).
set -uo pipefail
cd "$(dirname "$0")"

SPEC="cypress/e2e/container-journey.cy.js"
run_variant() { CYPRESS_CONTAINER="$1" npx cypress run --spec "$SPEC"; }

if [ "${1:-}" = "--parallel" ]; then
  echo "[suite] running all variants SIMULTANEOUSLY (output → <variant>.run.log)…"
  run_variant openai    >openai.run.log    2>&1 & pid_openai=$!
  run_variant anthropic >anthropic.run.log 2>&1 & pid_anthropic=$!
  run_variant qwen      >qwen.run.log      2>&1 & pid_qwen=$!
  wait "$pid_openai";    rc_openai=$?
  wait "$pid_anthropic"; rc_anthropic=$?
  wait "$pid_qwen";      rc_qwen=$?
  echo "[suite] openai    → rc=$rc_openai    (see openai.run.log)"
  echo "[suite] anthropic → rc=$rc_anthropic (see anthropic.run.log)"
  echo "[suite] qwen      → rc=$rc_qwen      (see qwen.run.log)"
else
  echo "[suite] running variants SEQUENTIALLY: openai, anthropic, qwen…"
  run_variant openai;    rc_openai=$?
  run_variant anthropic; rc_anthropic=$?
  run_variant qwen;      rc_qwen=$?
  echo "[suite] openai → rc=$rc_openai    anthropic → rc=$rc_anthropic    qwen → rc=$rc_qwen"
fi

if [ "$rc_openai" -eq 0 ] && [ "$rc_anthropic" -eq 0 ] && [ "$rc_qwen" -eq 0 ]; then
  echo "[suite] ✓ all variants passed"
  exit 0
fi
echo "[suite] ✗ a variant failed (openai=$rc_openai anthropic=$rc_anthropic qwen=$rc_qwen)"
exit 1
