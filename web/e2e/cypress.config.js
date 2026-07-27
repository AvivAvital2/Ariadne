const { defineConfig } = require('cypress');
const { spawn } = require('node:child_process');
const net = require('node:net');
const http = require('node:http');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

// Repo root (this file lives at web/e2e/cypress.config.js).
const REPO_ROOT = path.resolve(__dirname, '..', '..');
const HOST = '127.0.0.1';

// Track EVERY spawned server so none can leak. A single missed teardown, a
// second startAriadne before a stop, or a Cypress crash/Ctrl-C used to leave
// DETACHED `ariadne serve` processes alive — and they piled up by the thousand.
// We SIGKILL the whole process group (incl. the MCP child) and reap all
// survivors when Cypress itself exits.
const servers = new Set();   // each: { proc, workspace }
let current = null;          // the server for the test in flight

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
    // NOTE: no baseUrl on purpose. Cypress verifies a configured baseUrl at
    // launch (before any hook) — that would fail because each spec/test boots
    // its own server. Specs hold the baseUrl in a closure var, set from the
    // startAriadne task, and use absolute URLs.
    specPattern: 'cypress/e2e/**/*.cy.js',
    supportFile: 'cypress/support/e2e.js',
    fixturesFolder: false,
    video: true,               // record each spec → cypress/videos/*.mp4 to review the hang
    pageLoadTimeout: 120000,   // 2-min cap on cy.visit (a stuck load fails + saves the video)
    defaultCommandTimeout: 8000,
    taskTimeout: 120000,       // 2-min cap on cy.task (server start/stop)
    experimentalMemoryManagement: true,   // reduce run-mode renderer memory pressure
    setupNodeEvents(on) {
      // Headless Electron crashes the renderer on GPU-composited CSS (the
      // topbar's backdrop-filter:blur), and in `cypress run` a renderer crash
      // RESTARTS the whole spec — which re-runs before() → startAriadne, over
      // and over (that's the "214 boots"). Force software/CPU compositing so
      // the blur can't take the GPU process down. Fixes the browser, not the page.
      on('before:browser:launch', (browser, launchOptions) => {
        if (browser.family === 'chromium') {
          launchOptions.args.push('--disable-gpu', '--disable-gpu-compositing', '--disable-dev-shm-usage');
        }
        return launchOptions;
      });
      on('task', {
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
