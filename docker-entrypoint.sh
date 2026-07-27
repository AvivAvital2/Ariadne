#!/bin/sh
# Ariadne container entrypoint — the "both doors, one brain" topology.
#
#   agents  ──▶ :8000/mcp  ariadne mcp --http   (the single MCP brain)
#   browser ──▶ :8765      ariadne serve  ──(MCP client via --mcp-url)──┘
#
# The web UI is a pure MCP client of the brain, so there is exactly one process
# writing the DB (the brain) — see the single-writer rule in the blueprint.
set -eu

BRAIN_HOST=0.0.0.0     # bind 0.0.0.0 so the host port-mapping can reach it, and
BRAIN_PORT=8000        # so FastMCP's Host-allowlist (127.0.0.1/localhost only) is
WEB_HOST=0.0.0.0       # not auto-triggered (mcp server.py:177-183).
WEB_PORT=8765
MCP_URL="http://127.0.0.1:${BRAIN_PORT}/mcp"   # serve path IS /mcp (FastMCP default)

# Verbose brain/web logging when ARIADNE_DEBUG is set (the e2e harness sets it),
# so onboard progress + errors reach stdout / `docker logs`. Off by default so a
# normal `docker compose up` stays quiet. --debug is a GLOBAL flag (cli/main.py).
DEBUG_FLAG=""
if [ -n "${ARIADNE_DEBUG:-}" ]; then DEBUG_FLAG="--debug"; fi

# 0) First-run bootstrap. Ensure the config file exists at ARIADNE_CONFIG so it
#    is loaded FIRST (config.py search order: ARIADNE_CONFIG → cwd → repo → home)
#    and source_add/onboard persist to the /data volume — not to the repo's own
#    /opt/ariadne/ariadne.yaml. A missing file otherwise silently falls back to
#    defaults (config.py:315-320) and later writes can land off the volume.
: "${ARIADNE_CONFIG:=/data/ariadne.yaml}"
mkdir -p "$(dirname "${ARIADNE_CONFIG}")"
[ -f "${ARIADNE_CONFIG}" ] || printf 'sources: {}\n' > "${ARIADNE_CONFIG}"

# 1) Start the brain in the background.
echo "[ariadne] starting MCP brain on ${BRAIN_HOST}:${BRAIN_PORT} ..."
ariadne ${DEBUG_FLAG} mcp --http --host "${BRAIN_HOST}" --port "${BRAIN_PORT}" &
BRAIN_PID=$!

# If the brain dies, take the container down with it (no supervisor here; a
# process supervisor like s6/supervisord is a later hardening step).
trap 'kill "${BRAIN_PID}" 2>/dev/null || true' EXIT

# 2) Readiness gate. There is NO health endpoint, and the web's connect_http has
#    NO retry (web/mcp_client.py) — booting the web before the brain answers
#    would crash on_startup. Any HTTP response from /mcp means the session
#    manager is up; connection-refused means it isn't yet.
echo "[ariadne] waiting for the brain at ${MCP_URL} ..."
i=0
until curl -s -o /dev/null "${MCP_URL}"; do
    i=$((i + 1))
    if ! kill -0 "${BRAIN_PID}" 2>/dev/null; then
        echo "[ariadne] brain exited during startup — aborting." >&2
        exit 1
    fi
    if [ "${i}" -ge 120 ]; then   # ~60s
        echo "[ariadne] brain did not become ready in time — aborting." >&2
        exit 1
    fi
    sleep 0.5
done
echo "[ariadne] brain ready."

# 3) Start the web UI in the foreground (it becomes the container's main process).
echo "[ariadne] starting web UI on ${WEB_HOST}:${WEB_PORT} → ${MCP_URL}"
exec ariadne ${DEBUG_FLAG} serve --host "${WEB_HOST}" --port "${WEB_PORT}" --mcp-url "${MCP_URL}"
