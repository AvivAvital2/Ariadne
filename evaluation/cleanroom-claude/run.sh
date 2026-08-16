#!/usr/bin/env bash
# Build the hermetic image (which fetches the pinned UPSTREAM source at build
# time) and run a fresh Claude Code over the questions.
#
#   ANTHROPIC_API_KEY=sk-... ./run.sh
#
# The corpus is baked into the image from upstream tags — there is NO host
# filesystem in play: no /corpus mount, no local spool-corpus, no cleaning
# step. The ONLY mount is /work, a scratch dir carrying the stripped questions
# in and the answers out (never any corpus, never anything Ariadne-side).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY}"

# Which question set the bare arm answers. Defaults to the DE-BREADCRUMBED set.
# questions_25_clean.json states the answer inside the question: all 25 of those
# prompts name the Delta/Spark class outright ("which Spark class does
# DeltaFileFormatWriter reimplement as a fork"), 101 symbol mentions across the
# set, so both arms score full marks and the head-to-head ties 25/25 at 9.00 —
# no discriminating power at all. Container isolation cannot help with a leak
# that is in the prompt. Ariadne is measured on the de-breadcrumbed wording, so
# the bare arm has to face that same wording or the two are sitting different
# exams. Pass a path to override.
QUESTIONS="${1:-$HERE/../spool-clean-room/questions_debcrumb_ask.json}"
[ -f "$QUESTIONS" ] || { echo "no such question file: $QUESTIONS" >&2; exit 1; }
OUT="$HERE/answers-cleanroom-$(basename "$QUESTIONS" .json).json"

# Build fetches spark/delta/sdk from upstream at the pinned tags (see Dockerfile).
docker build -t cleanroom-claude "$HERE"

# /work = stripped questions in, answers out. Nothing else is mounted.
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cp "$QUESTIONS" "$WORK/questions.json"

docker run --rm \
  -e ANTHROPIC_API_KEY \
  -e CONCURRENCY \
  -e MAX_AGENT_TURNS="${MAX_AGENT_TURNS:-20}" \
  -e MAX_COST_USD="${MAX_COST_USD:-2.0}" \
  -v "$WORK":/work \
  cleanroom-claude

cp "$WORK/answers.json" "$OUT"
echo "clean-room answers -> $OUT"
