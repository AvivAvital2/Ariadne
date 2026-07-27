const { defineConfig } = require('cypress');
const { spawn } = require('node:child_process');
const net = require('node:net');
const http = require('node:http');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

// Repo root (this file lives at web/e2e/cypress.config.js).
const REPO_ROOT = path.resolve(__dirname, '..', '..');
// Use `localhost` (NOT 127.0.0.1) — bind, readiness ping, and the URL handed to
// Cypress. Cypress runs its own proxy on `localhost`; visiting 127.0.0.1 is a
// DIFFERENT super-domain (cross-origin), which breaks cy.visit's socket
// injection (CDP -32000) so Cypress re-runs the spec forever — the boot loop.
// EVIDENCED headed: with 127.0.0.1, `cypress open` loops (server boots ready in
// 2.4s, Cypress reloads the spec, repeat); with `localhost` the tests actually
// run. `--host localhost` binds both 127.0.0.1 and ::1 so the browser reaches
// it. (Headless `cypress run` still loops for a separate, unsolved reason.)
const HOST = 'localhost';

// Track EVERY spawned server so none can leak. A single missed teardown, a
// second startAriadne before a stop, or a Cypress crash/Ctrl-C used to leave
// DETACHED `ariadne serve` processes alive — and they piled up by the thousand.
// We SIGKILL the whole process group (incl. the MCP child) and reap all
// survivors when Cypress itself exits.
const servers = new Set();   // each: { proc, workspace }
let current = null;          // the server for the test in flight

// A PLAIN Node static file server for the UI specs, booted ONCE per run (see
// setupNodeEvents) and served at a FIXED URL. This exists to fix the old
// `cypress run` boot-loop: the specs used to boot a fresh `ariadne serve` per
// spec, so when Cypress re-attempts a spec the server re-booted on a NEW port —
// cy.visit chased a moving origin and Cypress's CDP execution context never
// settled (-32000 "Cannot find context", retried forever). Visiting one fixed
// URL recovers cleanly, exactly like any external cy.visit. The specs stub every
// /api, so they only need the page served; the real backend is exercised by
// backend-atomicity.cy.js via cy.request (no page visit → no injection).
const WEB_STATIC = path.join(REPO_ROOT, 'web', 'static');
const staticServers = new Set();
const CTYPE = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.json': 'application/json', '.ico': 'image/x-icon' };

function startStaticServer(host = HOST) {
  return new Promise((resolve, reject) => {
    const srv = http.createServer((req, res) => {
      let p = decodeURIComponent((req.url || '/').split('?')[0]);
      if (p === '/') p = '/onboarding.html';
      p = p.replace(/^\/static\//, '/');
      const fp = path.join(WEB_STATIC, path.normalize(p));
      if (!fp.startsWith(WEB_STATIC) || !fs.existsSync(fp) || !fs.statSync(fp).isFile()) {
        res.writeHead(404, { 'Content-Type': 'text/plain' }); res.end('not found'); return;
      }
      res.writeHead(200, { 'Content-Type': CTYPE[path.extname(fp)] || 'application/octet-stream' });
      res.end(fs.readFileSync(fp));   // plain: no ETag/Range/conditional/sendfile
    });
    srv.on('error', reject);
    srv.listen(0, host, () => {
      staticServers.add(srv);
      resolve({ baseUrl: `http://${host}:${srv.address().port}` });
    });
  });
}

function closeStaticServers() {
  for (const s of staticServers) { try { s.close(); } catch (e) { /* gone */ } }
  staticServers.clear();
}

// --- Journey test support -------------------------------------------------
// ONE persistent REAL `ariadne serve` (fixed 127.0.0.1 URL, so cy.visit survives
// a re-attempt) on a workspace holding a small synthetic source with clear,
// answerable content. Cached — booted once. The journey does a REAL generate +
// REAL ask, so the run env MUST have the LLM/embedding API keys.
let journeyProc = null;
let journey = null;
const RETRY_PY = [
  '"""Retry helpers: wrap flaky calls in exponential backoff."""',
  'import time',
  '',
  '',
  'def retry_with_backoff(fn, attempts=3, base_delay=0.1):',
  '    """Call ``fn``, retrying up to ``attempts`` times with exponential backoff.',
  '',
  '    After each failure the delay doubles (base_delay, 2x, 4x, ...). The last',
  '    exception is re-raised once all attempts are exhausted."""',
  '    delay = base_delay',
  '    for attempt in range(attempts):',
  '        try:',
  '            return fn()',
  '        except Exception:',
  '            if attempt == attempts - 1:',
  '                raise',
  '            time.sleep(delay)',
  '            delay *= 2',
  '',
].join('\n');

// A tiny Go module alongside RETRY_PY, so the container source is MULTI-LANGUAGE
// (Python + Go) → the in-container onboard runs two SCIP indexers and the real
// `scip merge` actually combines them (single-language skips merge). go.mod pins
// a version the image's Go toolchain (1.22.x) satisfies; stdlib-only so scip-go
// needs no network.
const RETRY_GO_MOD = 'module retryutils\n\ngo 1.22\n';
const RETRY_GO = [
  '// Package retryutils retries flaky calls with exponential backoff.',
  'package retryutils',
  '',
  'import "time"',
  '',
  '// RetryWithBackoff calls fn, retrying up to attempts times with exponential',
  '// backoff (base, 2x, 4x, ...). The last error is returned once attempts run out.',
  'func RetryWithBackoff(fn func() error, attempts int, base time.Duration) error {',
  '    var err error',
  '    delay := base',
  '    for i := 0; i < attempts; i++ {',
  '        if err = fn(); err == nil {',
  '            return nil',
  '        }',
  '        if i < attempts-1 {',
  '            time.Sleep(delay)',
  '            delay *= 2',
  '        }',
  '    }',
  '    return err',
  '}',
  '',
].join('\n');

function pingHost(host, port) {
  return new Promise((resolve) => {
    const req = http.get({ host, port, path: '/' }, (res) => { res.resume(); resolve(true); });
    req.on('error', () => resolve(false));
    req.setTimeout(1500, () => { req.destroy(); resolve(false); });
  });
}

async function bootJourneyServer() {
  if (journey) return journey;                       // cached → fixed URL across re-attempts
  const jhost = '127.0.0.1';
  const port = await new Promise((resolve, reject) => {
    const s = net.createServer();
    s.on('error', reject);
    s.listen(0, jhost, () => { const p = s.address().port; s.close(() => resolve(p)); });
  });
  const workspace = fs.mkdtempSync(path.join(os.tmpdir(), 'ariadne-journey-'));
  const configPath = path.join(workspace, 'ariadne.yaml');
  fs.writeFileSync(configPath, 'sources: {}\n');
  const sourcePath = path.join(workspace, 'retryutils');
  fs.mkdirSync(sourcePath, { recursive: true });
  fs.writeFileSync(path.join(sourcePath, 'retry.py'), RETRY_PY);

  console.log(`[journey] booting REAL ariadne serve on http://${jhost}:${port} (cwd=${REPO_ROOT})`);
  const proc = spawn('uv', ['run', 'ariadne', 'serve', '--host', jhost, '--port', String(port)], {
    cwd: REPO_ROOT,
    env: { ...process.env, ARIADNE_CONFIG: configPath },
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: true,
  });
  journeyProc = proc;
  let out = '';
  const grab = (d) => { out = (out + d).slice(-8000); process.stdout.write('[journey-serve] ' + d); };
  proc.stdout.on('data', grab);
  proc.stderr.on('data', grab);

  const deadline = Date.now() + 90000;
  while (Date.now() < deadline) {
    if (await pingHost(jhost, port)) {
      journey = { baseUrl: `http://${jhost}:${port}`, sourceName: 'retryutils', sourcePath, configPath, workspace };
      console.log(`[journey] server ready: ${journey.baseUrl}`);
      return journey;
    }
    await new Promise((r) => setTimeout(r, 300));
  }
  throw new Error('[journey] ariadne serve not ready after 90s\n--- output ---\n' + (out || '(none)'));
}

// A REAL `ariadne generate` against the journey source (same config/DB the
// journey server reads) so the ask has genuine docs. --yes = non-interactive.
function journeyGenerate({ sourceName, configPath }) {
  return new Promise((resolve) => {
    console.log(`[journey] REAL generate --source ${sourceName}`);
    const proc = spawn('uv', ['run', 'ariadne', 'generate', '--source', sourceName, '--yes'], {
      cwd: REPO_ROOT,
      env: { ...process.env, ARIADNE_CONFIG: configPath },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let out = '';
    const grab = (d) => { out = (out + d).slice(-16000); process.stdout.write('[journey-gen] ' + d); };
    proc.stdout.on('data', grab);
    proc.stderr.on('data', grab);
    proc.on('exit', (code) => resolve({ rc: code, output: out.slice(-4000) }));
  });
}

// --- Container e2e support -------------------------------------------------
// Boots the REAL image via `docker compose` (the actual deploy path) and drives
// the full journey against it — the automated Phase-0 acceptance the blueprint
// left manual. Layers web/e2e/compose.e2e.yaml over the real compose.yaml (small
// Python+Go workspace so `scip merge` runs, distinct ports, verbose, JVM-less).
// Gated (CYPRESS_CONTAINER=1); needs a reachable Docker daemon + LLM/embedding
// keys. Can't run in the author's sandbox (docker socket blocked) — you run it.
// Per-variant isolation so the two journeys can run SIMULTANEOUSLY without
// colliding: each boot gets its own compose PROJECT, host PORTS, and image TAG
// (passed from the spec's VARIANTS, threaded to compose via env vars below).
// --project-directory so `build: context: .` + `env_file: .env` resolve to the
// repo root regardless of the cypress cwd.
const COMPOSE_FILES = [
  '-f', path.join(REPO_ROOT, 'compose.yaml'),
  '-f', path.join(REPO_ROOT, 'web', 'e2e', 'compose.e2e.yaml'),
  '--project-directory', REPO_ROOT];
const composeArgs = (project) => ['compose', '-p', project, ...COMPOSE_FILES];
let currentProject = null;      // compose project of the live stack (for teardown)
let containerWorkspace = null;
let containerBooted = null;
let containerLogProc = null;    // `docker logs -f` follower — streams the in-container onboard
let containerHeartbeat = null;  // elapsed-time ticker so a quiet onboard never looks frozen
let containerTornDown = false;  // set by containerDown so reapAll doesn't re-`docker logs` a gone container (clobbering the saved log)

function sh(cmd, args, opts = {}) {
  return new Promise((resolve) => {
    const p = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'], ...opts });
    let out = '';
    const grab = (d) => { out = (out + d).slice(-16000); if (opts.echo) process.stdout.write(d); };
    p.stdout.on('data', grab);
    p.stderr.on('data', grab);
    p.on('exit', (code) => resolve({ rc: code, out }));
    p.on('error', (e) => resolve({ rc: -1, out: String((e && e.message) || e) }));
  });
}

// Ariadne reads API keys from the repo-root .env (python-dotenv), so they're
// usually NOT exported in the shell — but `docker run -e KEY` and the preflight
// below only see EXPORTED vars. Read that .env as a fallback so the container
// gets the same keys Ariadne would, with no manual `export`. (Values only, never
// logged; a shell-exported var still wins.)
function keyFromDotenv(name) {
  let text;
  try { text = fs.readFileSync(path.join(REPO_ROOT, '.env'), 'utf8'); } catch (e) { return ''; }
  for (let raw of text.split(/\r?\n/)) {
    let line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    if (line.startsWith('export ')) line = line.slice(7).trim();
    const eq = line.indexOf('=');
    if (eq === -1 || line.slice(0, eq).trim() !== name) continue;
    let v = line.slice(eq + 1).trim();
    if (v.length >= 2 && ((v[0] === '"' && v.endsWith('"')) || (v[0] === "'" && v.endsWith("'")))) {
      v = v.slice(1, -1);
    }
    return v;
  }
  return '';
}

async function bootContainer(opts = {}) {
  if (containerBooted) return containerBooted;                    // cached across re-attempts
  containerTornDown = false;                                      // fresh boot (variants boot sequentially)

  // Preflight (fail fast + free): OPENAI_API_KEY is ALWAYS required — embeddings +
  // search are OpenAI-only regardless of the generation provider (Anthropic has no
  // embeddings API). ANTHROPIC_API_KEY is required only for the Claude-generation
  // variant. Resolve from the shell env or the repo .env, and populate process.env
  // so the compose `environment:` passthrough forwards the value into the container.
  const required = ['OPENAI_API_KEY'];
  if (opts.requireAnthropic) required.push('ANTHROPIC_API_KEY');
  for (const k of required) {
    const v = process.env[k] || keyFromDotenv(k);
    if (!v) {
      throw new Error(`[container] ${k} not found in the shell env or ${REPO_ROOT}/.env — set it in one of those. `
        + (opts.requireAnthropic
          ? 'The Claude-generation variant needs BOTH keys (OPENAI for embeddings, ANTHROPIC for generation).'
          : 'The all-OpenAI variant needs OPENAI_API_KEY (generation + embeddings).'));
    }
    process.env[k] = v;
  }

  const ver = await sh('docker', ['version', '--format', '{{.Server.Version}}']);
  if (ver.rc !== 0) throw new Error('[container] docker daemon not reachable:\n' + ver.out);

  // Per-variant isolation (so two runs can go simultaneously): project, ports,
  // and image tag all come from the caller (the spec's VARIANTS).
  const project = opts.project || 'ariadne-e2e';
  const webPort = opts.webPort || 18765;
  currentProject = project;
  const COMPOSE = composeArgs(project);

  // Small MULTI-LANGUAGE source (Python + Go) so the in-container onboard runs
  // TWO SCIP indexers and the real `scip merge` actually combines them (a
  // single-language source would skip merge — see cli/index.py:798).
  containerWorkspace = fs.mkdtempSync(path.join(os.tmpdir(), `ariadne-container-${project}-`));
  const src = path.join(containerWorkspace, 'retryutils');
  fs.mkdirSync(src, { recursive: true });
  fs.writeFileSync(path.join(src, 'retry.py'), RETRY_PY);
  fs.writeFileSync(path.join(src, 'go.mod'), RETRY_GO_MOD);
  fs.writeFileSync(path.join(src, 'retry.go'), RETRY_GO);
  // Threaded into compose.e2e.yaml via ${...}: the workspace mount, the distinct
  // host ports, and a per-variant image tag (so parallel builds don't clash).
  process.env.E2E_WORKSPACE = containerWorkspace;
  process.env.E2E_WEB_PORT = String(webPort);
  process.env.E2E_MCP_PORT = String(opts.mcpPort || 18000);
  process.env.E2E_VARIANT = opts.variantKey || 'default';

  // Bring the stack up via docker compose (the real deploy path): base
  // compose.yaml + the e2e override. --build so a Dockerfile change is never
  // masked by a stale image; Docker's layer cache keeps an unchanged build fast.
  console.log(`[${project}] docker compose up --build → http://127.0.0.1:${webPort}`);
  await sh('docker', [...COMPOSE, 'down', '-v', '--remove-orphans']);   // clear any stale stack for THIS project
  const up = await sh('docker', [...COMPOSE, 'up', '-d', '--build'], { cwd: REPO_ROOT, echo: true });
  if (up.rc !== 0) throw new Error(`[${project}] docker compose up FAILED:\n` + up.out.slice(-6000));

  // VISIBILITY: stream the stack's logs live + a heartbeat with elapsed time, so
  // the slow, quiet in-container onboard never looks frozen at "ready". The
  // per-project prefix keeps simultaneous variants distinguishable.
  containerLogProc = spawn('docker', [...COMPOSE, 'logs', '-f', '--tail', '20'],
    { stdio: ['ignore', 'pipe', 'pipe'] });
  const relay = (d) => process.stdout.write(`[${project}] ` + d);
  containerLogProc.stdout.on('data', relay);
  containerLogProc.stderr.on('data', relay);
  const startedAt = Date.now();
  containerHeartbeat = setInterval(() => {
    process.stdout.write(`[${project}] still working — ${Math.round((Date.now() - startedAt) / 1000)}s elapsed `
      + `(real onboard/ask; \`docker compose -p ${project} logs -f\` for detail)\n`);
  }, 20000);

  // Readiness: the web answers `/` only after the brain is up (entrypoint gates).
  const deadline = Date.now() + 240000;
  while (Date.now() < deadline) {
    if (await pingHost('127.0.0.1', webPort)) {
      containerBooted = {
        baseUrl: `http://127.0.0.1:${webPort}`,
        sourceName: 'retryutils',
        sourcePath: '/workspace/retryutils',                     // container-internal path (the mount)
        workspace: containerWorkspace,
      };
      console.log(`[${project}] ready: ${containerBooted.baseUrl}`);
      return containerBooted;
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
  const logs = await sh('docker', [...COMPOSE, 'logs', '--tail', '150']);
  throw new Error(`[${project}] not ready after 240s:\n` + logs.out);
}

function stopContainerLogs() {
  if (containerHeartbeat) { try { clearInterval(containerHeartbeat); } catch (e) { /* gone */ } containerHeartbeat = null; }
  if (containerLogProc && containerLogProc.pid) { try { containerLogProc.kill('SIGKILL'); } catch (e) { /* gone */ } containerLogProc = null; }
}

async function containerDown() {
  stopContainerLogs();
  const project = currentProject || 'ariadne-e2e';
  const COMPOSE = composeArgs(project);
  // Preserve this stack's FULL logs before tearing down (a failed onboard stays
  // diagnosable after the run). Per-project file so simultaneous variants don't
  // clobber each other.
  const dump = await sh('docker', [...COMPOSE, 'logs', '--no-color']);
  const logPath = path.join(__dirname, `${project}.log`);
  try { fs.writeFileSync(logPath, dump.out || '(no container logs captured)'); console.log(`[${project}] saved logs → ${logPath}`); } catch (e) { /* best effort */ }
  await sh('docker', [...COMPOSE, 'down', '-v', '--remove-orphans']);
  if (containerWorkspace) { try { fs.rmSync(containerWorkspace, { recursive: true, force: true }); } catch (e) { /* best effort */ } }
  containerWorkspace = null;
  containerBooted = null;
  containerTornDown = true;   // reapAll must not re-capture logs from the now-gone container
  return null;
}

function killServer(s) {
  if (!s) return;
  if (s.proc && s.proc.pid) {
    try { process.kill(-s.proc.pid, 'SIGKILL'); }        // whole group
    catch (e) { try { s.proc.kill('SIGKILL'); } catch (_) { /* gone */ } }
  }
  if (s.workspace) {
    try { fs.rmSync(s.workspace, { recursive: true, force: true }); } catch (e) { /* best effort */ }
  }
  servers.delete(s);
}

// Reap ALL servers if Cypress exits for ANY reason — normal end, crash, or
// Ctrl-C. This is what stops leaks when a run is interrupted mid-spec.
let reaped = false;
function reapAll() {
  if (reaped) return;
  reaped = true;
  for (const s of servers) {
    if (s.proc && s.proc.pid) { try { process.kill(-s.proc.pid, 'SIGKILL'); } catch (e) { /* gone */ } }
    if (s.workspace) { try { fs.rmSync(s.workspace, { recursive: true, force: true }); } catch (e) { /* best effort */ } }
  }
  servers.clear();
  closeStaticServers();
  if (journeyProc && journeyProc.pid) { try { process.kill(-journeyProc.pid, 'SIGKILL'); } catch (e) { /* gone */ } }
  // Reap the e2e compose stack + its temp workspace (best-effort, synchronous).
  // Skip entirely if containerDown already tore it down — otherwise `logs` on the
  // gone stack writes an error over the saved log.
  stopContainerLogs();
  if (!containerTornDown && currentProject) {
    const cf1 = path.join(REPO_ROOT, 'compose.yaml');
    const cf2 = path.join(REPO_ROOT, 'web', 'e2e', 'compose.e2e.yaml');
    const base = `docker compose -p ${currentProject} -f "${cf1}" -f "${cf2}" --project-directory "${REPO_ROOT}"`;
    try { require('node:child_process').execSync(`${base} logs --no-color > "${path.join(__dirname, currentProject + '.log')}" 2>&1`, { stdio: 'ignore' }); } catch (e) { /* stack gone / never started */ }
    try { require('node:child_process').execSync(`${base} down -v --remove-orphans`, { stdio: 'ignore' }); } catch (e) { /* daemon down / never started */ }
  }
  if (containerWorkspace) { try { fs.rmSync(containerWorkspace, { recursive: true, force: true }); } catch (e) { /* best effort */ } }
}
process.on('exit', reapAll);
process.on('SIGINT', () => { reapAll(); process.exit(130); });
process.on('SIGTERM', () => { reapAll(); process.exit(143); });

function freePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.on('error', reject);
    srv.listen(0, HOST, () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

function ping(port) {
  return new Promise((resolve) => {
    const req = http.get({ host: HOST, port, path: '/' }, (res) => { res.resume(); resolve(true); });
    req.on('error', () => resolve(false));
    req.setTimeout(1500, () => { req.destroy(); resolve(false); });
  });
}

module.exports = defineConfig({
  e2e: {
    // No baseUrl: the UI specs (console/onboarding) visit the persistent static
    // server via the `fixedStaticUrl` task; backend-atomicity boots a per-test
    // `ariadne serve` and drives it with cy.request (no page visit). Both use
    // absolute URLs, so a configured baseUrl (which Cypress verifies at launch)
    // isn't needed.
    specPattern: 'cypress/e2e/**/*.cy.js',
    supportFile: 'cypress/support/e2e.js',
    fixturesFolder: false,
    video: true,               // record each spec → cypress/videos/*.mp4 to review the hang
    pageLoadTimeout: 120000,   // 2-min cap on cy.visit (a stuck load fails + saves the video)
    defaultCommandTimeout: 8000,
    taskTimeout: 120000,       // 2-min cap on cy.task (server start/stop)
    async setupNodeEvents(on) {
      // ONE persistent static server for the whole run (fixed port/URL), so a
      // spec re-run visits the SAME origin instead of a freshly-booted one.
      const fixed = await startStaticServer('127.0.0.1');
      on('task', {
        // Print a message to the terminal (used to surface test failures +
        // network signals where the Cypress UI command log isn't visible).
        log(message) {
          console.log(String(message));
          return null;
        },
        // The fixed server's base URL — same every re-run (no moving target).
        fixedStaticUrl() {
          return fixed.baseUrl;
        },
        // Journey test: boot the persistent REAL ariadne serve (cached) + run a
        // REAL generate. Both require LLM/embedding keys in the run env.
        async journeyServer() {
          return bootJourneyServer();
        },
        journeyGenerate(args) {
          return journeyGenerate(args);
        },

        // Container e2e: build + run the REAL stack (compose); tear it down.
        // opts.requireAnthropic gates the ANTHROPIC key preflight (Claude variant).
        async containerUp(opts) {
          return bootContainer(opts || {});
        },
        async containerDown() {
          return containerDown();
        },

        // Boot a FRESH, isolated Ariadne: a new temp config dir (empty DB +
        // empty cache) on a fresh port, plus a synthetic source directory for
        // real-backend specs. No state leaks between tests.
        // Returns { baseUrl, sourcePath }.
        async startAriadne() {
          // Defensive: if a prior server wasn't stopped (e.g. a spec used
          // before() and a second start slipped in), kill it first. A single
          // `proc` variable silently orphaned the previous server here.
          if (current) killServer(current);

          const port = await freePort();
          const workspace = fs.mkdtempSync(path.join(os.tmpdir(), 'ariadne-e2e-'));
          fs.writeFileSync(path.join(workspace, 'ariadne.yaml'), 'sources: {}\n');
          // a tiny real project for source_add/discover/estimate to operate on
          const sourcePath = path.join(workspace, 'proj');
          fs.mkdirSync(path.join(sourcePath, 'pkg'), { recursive: true });
          fs.writeFileSync(path.join(sourcePath, 'pkg', 'a.py'), 'def a():\n    return 1\n');

          // Capture AND live-stream the server's output to the terminal, so you
          // can see whether ariadne serve is booting or silent. Both streams are
          // consumed, so the child can never block on a full pipe buffer.
          console.log(`[e2e] booting: uv run ariadne serve --host ${HOST} --port ${port}  (cwd=${REPO_ROOT})`);
          const t0 = Date.now();
          let out = '';
          let exited = null;
          const proc = spawn('uv', ['run', 'ariadne', 'serve', '--host', HOST, '--port', String(port)], {
            cwd: REPO_ROOT,
            env: { ...process.env, ARIADNE_CONFIG: path.join(workspace, 'ariadne.yaml') },
            stdio: ['ignore', 'pipe', 'pipe'],
            detached: true,   // own process group → clean group kill (incl. the MCP child)
          });
          const s = { proc, workspace };
          servers.add(s);
          current = s;
          proc.on('exit', (code, signal) => { exited = { code, signal }; servers.delete(s); });

          const grab = (d) => { const t = d.toString(); out = (out + t).slice(-8000); process.stdout.write('[serve] ' + t); };
          proc.stdout.on('data', grab);
          proc.stderr.on('data', grab);

          // Poll for readiness with periodic progress logs; fail with the server's
          // output if it dies or never answers — so a stuck/broken server is
          // obvious in the terminal, not a silent hang.
          const deadline = Date.now() + 90000;
          let lastLog = 0;
          while (Date.now() < deadline) {
            if (exited) {
              killServer(s);
              throw new Error(
                `[e2e] ariadne serve EXITED before ready (code=${exited.code}, signal=${exited.signal}).\n`
                + `--- server output ---\n${out || '(no output)'}`);
            }
            if (await ping(port)) {
              console.log(`[e2e] server ready in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
              return { baseUrl: `http://${HOST}:${port}`, sourcePath };
            }
            const waited = Math.round((Date.now() - t0) / 1000);
            if (waited >= lastLog + 5) { lastLog = waited; console.log(`[e2e] waiting for server to answer /  … ${waited}s`); }
            await new Promise((r) => setTimeout(r, 300));
          }
          killServer(s);
          throw new Error(
            `[e2e] ariadne serve NOT ready after 90s on http://${HOST}:${port}.\n`
            + `--- server output (tail) ---\n${out || '(no output — is `uv run ariadne serve` runnable from the repo root?)'}`);
        },

        async stopAriadne() {
          killServer(current);
          current = null;
          return null;
        },
      });
    },
  },
});
