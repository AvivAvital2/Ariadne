# Ariadne — one fat image: the Python runtime + every SCIP indexer, so a single
# container can index arbitrary mounted projects. Phase 0 of the console blueprint
# (designs/web-ui/console-and-deployment.md). Authored but NOT build-tested here
# (no Docker daemon in the dev sandbox) — expect to iterate versions/URLs marked
# "VERIFY" on the first real `docker build`.
#
# Floor is Python >=3.12.6 (pyproject.toml). The author runs 3.14, but 3.14 has
# rough tooling edges (numpy/pytest-cov) — base on 3.12.
FROM python:3.12-slim-bookworm

# --- OS packages ------------------------------------------------------------
# tini: PID 1 signal forwarding (clean docker stop). build-essential+cmake: native
# wheels (hnswlib/leidenalg/igraph). openjdk-17: scip-java compiles JVM targets.
# nodejs/npm: scip-python + scip-typescript + the Vue extractor. git/curl: fetch
# tools, sync, and `spools create` clones.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates tini \
        build-essential cmake \
        openjdk-17-jdk-headless \
        nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# CRITICAL: scip-java's JDK auto-discovery globs are macOS-only
# (docgen/scip_indexers.py:905-948) — a Linux JDK is never found unless JAVA_HOME
# is set explicitly. Without this, Scala/Java indexing runs on the wrong/no JDK.
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# --- Go toolchain (apt's is too old for scip-go@latest) — official tarball ---
ENV GO_VERSION=1.22.5
RUN curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" | tar -C /usr/local -xz
ENV PATH="/usr/local/go/bin:/root/go/bin:${PATH}"

# --- SCIP indexers (none auto-install; each hard-fails if absent) -----------
# scip-python / scip-typescript via npm (docgen/scip_indexers.py:504,641)
RUN npm install -g @sourcegraph/scip-python @sourcegraph/scip-typescript

# scip-go via `go install` (docgen/scip_indexers.py:700).
# VERIFY the module org: the in-code hint says `scip-code/scip-go` but the
# canonical Sourcegraph module is `sourcegraph/scip-go` (likely a typo in the hint).
RUN go install github.com/sourcegraph/scip-go/cmd/scip-go@latest

# Coursier (cs) → scip-java + sbt. scip-java compiles the target via sbt/maven/
# gradle; sbt-first is required for Scala (maven emits Java-only SemanticDB).
# VERIFY the `cs install` flag name (--install-dir vs --dir) for the cs version.
RUN curl -fsSL "https://github.com/coursier/coursier/releases/latest/download/cs-x86_64-pc-linux.gz" \
        | gzip -d > /usr/local/bin/cs \
    && chmod +x /usr/local/bin/cs \
    && cs install --install-dir /usr/local/bin scip-java sbt

# scip merge CLI (cli/index.py:160-171) — only used across multi-language corpora.
# VERIFY the release asset name for your target arch.
RUN curl -fsSL "https://github.com/sourcegraph/scip/releases/latest/download/scip-linux-amd64.tar.gz" \
        | tar -xz -C /usr/local/bin scip

# --- Ariadne itself (reproducible from uv.lock) -----------------------------
RUN pip install --no-cache-dir uv
COPY . /opt/ariadne
WORKDIR /opt/ariadne
RUN uv sync --frozen --no-dev
# Put the project venv's `ariadne` console script on PATH.
ENV PATH="/opt/ariadne/.venv/bin:${PATH}"

# --- Runtime layout ---------------------------------------------------------
# ARIADNE_CONFIG anchors ALL Ariadne-owned data (DB, staleness DB, docs, run
# logs, spool store are config-dir-relative — config.py) under the /data volume.
ENV ARIADNE_CONFIG=/data/ariadne.yaml
RUN mkdir -p /data /workspace
WORKDIR /workspace

EXPOSE 8765 8000

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["tini", "--", "/usr/local/bin/docker-entrypoint.sh"]

# NOTE (uid): this image runs as root, so files Ariadne writes into /workspace
# (each project's .ariadne/, and build tools' .venv/target/) are root-owned on
# the host. For a single-user localhost box that's acceptable; matching the host
# uid (build arg + `useradd`) is a later hardening step.
