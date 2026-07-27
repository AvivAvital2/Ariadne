# syntax=docker/dockerfile:1
#
# Ariadne — one fat image: the Python runtime + every SCIP indexer, so a single
# container can index arbitrary mounted projects and serve both doors (browser +
# MCP-over-HTTP). Phase 0 of designs/web-ui/console-and-deployment.md.
#
# Build (amd64 host — Intel Mac / most CI):
#   docker build -t ariadne:latest .
# Options:
#   --build-arg WITH_JVM=0                     # slim image, no Scala/Java indexing
#   --build-arg UID=$(id -u) --build-arg GID=$(id -g)   # Linux host: own /workspace writes
#   --platform linux/amd64                     # on arm64, forces emulation (assets are amd64)
#
# Floor is Python >=3.12.6 (pyproject.toml); base on 3.12 (3.14 has rough
# numpy/pytest tooling edges). Toolchain downloads are amd64-pinned — the asset
# names marked VERIFY are the first things a real build confirms.
FROM python:3.12-slim-bookworm

# --- base OS packages (always) ----------------------------------------------
# tini: PID 1 signal forwarding (clean docker stop). build-essential+cmake:
# native wheels (hnswlib/leidenalg/igraph). git/curl/ca-certificates/gnupg:
# fetch tools, sync, spool clones, and the NodeSource key. (Node is installed
# separately below — bookworm's apt `nodejs` is the EOL 18.x line.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates gnupg tini \
        build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# --- Node.js 22 LTS (drives scip-python + scip-typescript, both Node CLIs) ---
# Debian bookworm's apt nodejs is 18.20.x (EOL Apr 2025); pull a current LTS from
# NodeSource so the JS/TS/py indexers run on a supported runtime (matches the
# host's Node 22). The NodeSource deb bundles npm.
ENV NODE_MAJOR=22
RUN curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# --- Go toolchain + scip-go (apt's go is too old for scip-go@latest) --------
# GOBIN=/usr/local/bin so the binary stays reachable after we drop to a non-root
# user (Go's default /root/go/bin is unreadable — /root is mode 0700).
# Module path is github.com/scip-code/scip-go — verified against the repo's
# go.mod (`module github.com/scip-code/scip-go`). The source lives at
# sourcegraph/scip-go but declares scip-code/scip-go as its module, so `go
# install` MUST use scip-code (matches the hint at docgen/scip_indexers.py:702).
ENV GO_VERSION=1.22.5
RUN curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" | tar -C /usr/local -xz
ENV PATH="/usr/local/go/bin:${PATH}"
RUN GOBIN=/usr/local/bin go install github.com/scip-code/scip-go/cmd/scip-go@latest

# --- SCIP indexers via npm (global bin is /usr/local/bin — reachable) -------
# docgen/scip_indexers.py:504,641 invoke `scip-python` / `scip-typescript`.
RUN npm install -g @sourcegraph/scip-python @sourcegraph/scip-typescript

# --- scip merge CLI — built from source with the vendored PR #420 command ----
# Ariadne runs `scip merge` (cli/index.py:161) to combine per-language indexes for
# multi-language sources. `merge` exists ONLY in scip-code/scip PR #420, which was
# never merged and ships in NO release — the stock `scip` binary has no `merge`.
# So we vendor the command (scripts/scip/merge.go) into the repo and build scip
# from source, registering it with a fail-loud literal replace (only needs the
# one distinctive commands() return line — robust to line numbers / whitespace).
# Pinned to commit e8ee0ae (scip-code/scip's release commit). Built as root but
# installed to /usr/local/bin (world-executable) so the non-root user can run it.
COPY scripts/scip/merge.go /tmp/scip-src/merge.go
RUN git clone https://github.com/scip-code/scip.git /tmp/scip \
    && cd /tmp/scip \
    && git checkout e8ee0ae \
    && cp /tmp/scip-src/merge.go cmd/scip/merge.go \
    && python3 -c "import pathlib; f=pathlib.Path('cmd/scip/main.go'); s=f.read_text(); a='return []*cli.Command{&lint, &print, &snapshot, &stats, &test, &convert}'; assert a in s, 'scip merge registration anchor not found in cmd/scip/main.go'; f.write_text(s.replace(a, 'merge := mergeCommand()\n\treturn []*cli.Command{&lint, &print, &snapshot, &stats, &test, &convert, &merge}'))" \
    && go build -o /usr/local/bin/scip ./cmd/scip \
    && cd / && rm -rf /tmp/scip /tmp/scip-src

# --- JVM toolchain (optional: Scala/Java indexing) — gated to slim the image -
# scip-java COMPILES the target via sbt/maven/gradle (docgen/scip_languages.py:87,
# can_index_standalone=False), pulling deps from the network at index time;
# sbt-first is required for Scala. WITH_JVM=0 drops all of it (JDK + coursier +
# scip-java + sbt) for a much smaller, JVM-less image.
# CRITICAL: scip-java's JDK auto-discovery globs are macOS-only
# (docgen/scip_indexers.py:905-948), so a Linux JDK is only found via the ambient
# JAVA_HOME — set it here. coursier asset `cs-x86_64-pc-linux.gz` + `--install-dir`
# verified against coursier v2.1.24 (no aarch64-linux launcher exists — an arm64
# build would need coursier.jar + java instead).
ARG WITH_JVM=1
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
RUN if [ "$WITH_JVM" = "1" ]; then set -eux; \
        apt-get update && apt-get install -y --no-install-recommends openjdk-17-jdk-headless; \
        rm -rf /var/lib/apt/lists/*; \
        curl -fsSL "https://github.com/coursier/coursier/releases/latest/download/cs-x86_64-pc-linux.gz" \
            | gzip -d > /usr/local/bin/cs; \
        chmod +x /usr/local/bin/cs; \
        # scip-java is in coursier's CONTRIB channel (io.get-coursier:apps-contrib),
        # NOT the default — `--contrib` is REQUIRED (it's additive, so sbt still
        # resolves from the default channel). Bare `cs install scip-java` fails.
        cs install --contrib --install-dir /usr/local/bin scip-java sbt; \
    else echo "WITH_JVM=0 — skipping JDK/coursier/scip-java/sbt (no Scala/Java indexing)"; fi

# --- Ariadne itself (reproducible from uv.lock) -----------------------------
# The repo's .python-version pins 3.14 (the author's dev interpreter), but the
# image builds on 3.12 (smoother numpy/tooling) and the lock's requires-python is
# >=3.12.6, so 3.12 is valid. Two guards, both load-bearing:
#  * --python /usr/local/bin/python3.12 forces the base image's 3.12, overriding
#    .python-version (which is also dockerignored so it never reaches the image).
#  * UV_PYTHON_DOWNLOADS=never + only-system stop uv fetching a MANAGED CPython
#    into /root/.local — that dir is 0700, so once we drop to USER app the venv's
#    python symlink is untraversable and `ariadne` dies with "Permission denied".
ENV UV_PYTHON_PREFERENCE=only-system UV_PYTHON_DOWNLOADS=never
RUN pip install --no-cache-dir uv
COPY . /opt/ariadne
WORKDIR /opt/ariadne
RUN uv sync --frozen --no-dev --python /usr/local/bin/python3.12
# Put the project venv's `ariadne` console script on PATH.
ENV PATH="/opt/ariadne/.venv/bin:${PATH}"

# --- Runtime layout + non-root user -----------------------------------------
# ARIADNE_CONFIG anchors ALL Ariadne-owned data (DB, staleness DB, docs, run
# logs, spool store are config-dir-relative — config.py) under the /data volume.
ENV ARIADNE_CONFIG=/data/ariadne.yaml
# Non-root runtime (hardening). Match the host uid via build args so files
# Ariadne + the SCIP indexers write into the bind-mounted /workspace (.ariadne/,
# and build tools' .venv/target/) are owned by you on the host, not root. On
# Docker Desktop (Mac) bind mounts are writable by any uid, so the default 1000
# is fine; on a Linux host pass --build-arg UID=$(id -u) GID=$(id -g).
ARG UID=1000
ARG GID=1000
ENV HOME=/home/app
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN set -eu; \
    chmod +x /usr/local/bin/docker-entrypoint.sh; \
    groupadd -g "$GID" app || true; \
    useradd -m -u "$UID" -g "$GID" -s /usr/sbin/nologin app || true; \
    mkdir -p /data /workspace; \
    chown -R "$UID:$GID" /data /workspace /opt/ariadne /home/app
USER app
WORKDIR /workspace

# Both doors: browser UI (8765) + MCP-over-HTTP for external agents (8000).
EXPOSE 8765 8000

# The web UI answers once the brain is up (the entrypoint gates on it), so a
# generous start-period covers brain+web boot. curl is present in the image.
HEALTHCHECK --start-period=120s --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8765/ >/dev/null || exit 1

ENTRYPOINT ["tini", "--", "/usr/local/bin/docker-entrypoint.sh"]
