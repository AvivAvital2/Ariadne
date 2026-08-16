# Clean-room bare-LLM arm: a fresh Claude Code instance whose corpus is fetched
# from UPSTREAM at the pinned release tags, baked into the image. The host
# filesystem is eliminated entirely — no /corpus mount, no local spool-corpus,
# nothing Ariadne-derived can reach it. The source is provably the canonical
# release, not a local derivation.
#
# dbr17.3-lts components (from spools.yaml): spark 4.0.0, delta 4.0.0, sdk 0.121.0.
FROM node:22-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates ripgrep python3 && rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code

# Pristine upstream source at the exact tags. Shallow (no history), and .git /
# agent-config stripped so the corpus is pure release source and nothing else.
# NOTE: tag spellings are the one thing not verifiable at author time (no net
# here) — if a clone fails, fix the tag below; it's the SOLE source of code.
RUN mkdir -p /corpus \
 && git clone --depth 1 --branch v4.0.0   https://github.com/apache/spark                 /corpus/spark \
 && git clone --depth 1 --branch v4.0.0   https://github.com/delta-io/delta               /corpus/delta \
 && git clone --depth 1 --branch v0.121.0 https://github.com/databricks/databricks-sdk-py /corpus/databricks-sdk-py \
 && find /corpus -depth -type d \( -name '.git' -o -name '.claude' -o -name '.github' \) -exec rm -rf {} + \
 && du -sh /corpus

RUN useradd -m runner
USER runner
ENV HOME=/home/runner \
    DISABLE_AUTOUPDATER=1 DISABLE_TELEMETRY=1 DISABLE_ERROR_REPORTING=1 \
    DISABLE_NON_ESSENTIAL_MODEL_CALLS=1
RUN mkdir -p /home/runner/.claude \
 && printf '{"mcpServers":{}}\n' > /home/runner/.claude.json \
 && printf '{"mcpServers":{},"enableAllProjectMcpServers":false}\n' \
      > /home/runner/.claude/settings.json

COPY --chown=runner:runner entrypoint.py /home/runner/entrypoint.py
ENTRYPOINT ["python3", "/home/runner/entrypoint.py"]
