#!/usr/bin/env bash
# Container-journey SUITE — drives BOTH variants of
# cypress/e2e/container-journey.cy.js:
#   • openai    — gpt generation + OpenAI embeddings    (one key:  OPENAI_API_KEY)
#   • anthropic — Claude generation + OpenAI embeddings  (both keys)
#
# Usage (from web/e2e/):
#   ./run-container-suite.sh              # SEQUENTIAL: openai, then anthropic
#   ./run-container-suite.sh --parallel   # SIMULTANEOUS (isolated project/ports/image)
#
# Each variant runs as its own `cypress run` with a distinct compose project,
# host ports, and image tag (see the spec's VARIANTS), so --parallel is
# collision-free. Runs BOTH even if one fails, then reports; exits non-zero if
# either failed.
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
  echo "[suite] running BOTH variants SIMULTANEOUSLY (output → openai.run.log / anthropic.run.log)…"
  run_variant openai    >openai.run.log    2>&1 & pid_openai=$!
  run_variant anthropic >anthropic.run.log 2>&1 & pid_anthropic=$!
  wait "$pid_openai";    rc_openai=$?
  wait "$pid_anthropic"; rc_anthropic=$?
  echo "[suite] openai    → rc=$rc_openai    (see openai.run.log)"
  echo "[suite] anthropic → rc=$rc_anthropic (see anthropic.run.log)"
else
  echo "[suite] running variants SEQUENTIALLY: openai, then anthropic…"
  run_variant openai;    rc_openai=$?
  run_variant anthropic; rc_anthropic=$?
  echo "[suite] openai → rc=$rc_openai    anthropic → rc=$rc_anthropic"
fi

if [ "$rc_openai" -eq 0 ] && [ "$rc_anthropic" -eq 0 ]; then
  echo "[suite] ✓ both variants passed"
  exit 0
fi
echo "[suite] ✗ a variant failed (openai=$rc_openai anthropic=$rc_anthropic)"
exit 1
