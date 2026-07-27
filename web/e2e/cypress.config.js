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
