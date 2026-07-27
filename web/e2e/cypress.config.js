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

// One live server per test, tracked here so afterEach can tear it down.
let proc = null;
let workspace = null;

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
    // launch (before any hook) — that would fail because each test boots its
    // own server. The support file sets baseUrl per test, after startup.
    specPattern: 'cypress/e2e/**/*.cy.js',
    supportFile: 'cypress/support/e2e.js',
    fixturesFolder: false,
    video: true,               // record each spec → cypress/videos/*.mp4 to review the hang
    pageLoadTimeout: 120000,   // 2-min cap on cy.visit (a stuck load fails + saves the video)
    defaultCommandTimeout: 8000,
    taskTimeout: 120000,       // 2-min cap on cy.task (server start/stop)
    setupNodeEvents(on) {
      on('task', {
        // Boot a FRESH, isolated Ariadne for one test: a new temp config dir
        // (empty DB + empty cache) on a fresh port, plus a synthetic source
        // directory for real-backend specs. No state leaks between tests.
        // Returns { baseUrl, sourcePath }.
        async startAriadne() {
          const port = await freePort();
          workspace = fs.mkdtempSync(path.join(os.tmpdir(), 'ariadne-e2e-'));
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
          proc = spawn('uv', ['run', 'ariadne', 'serve', '--host', HOST, '--port', String(port)], {
            cwd: REPO_ROOT,
            env: { ...process.env, ARIADNE_CONFIG: path.join(workspace, 'ariadne.yaml') },
            stdio: ['ignore', 'pipe', 'pipe'],
            detached: true,   // own process group → clean group kill (incl. the MCP child)
          });
          const grab = (d) => { const s = d.toString(); out = (out + s).slice(-8000); process.stdout.write('[serve] ' + s); };
          proc.stdout.on('data', grab);
          proc.stderr.on('data', grab);
          proc.on('exit', (code, signal) => { exited = { code, signal }; });

          // Poll for readiness with periodic progress logs; fail with the server's
          // output if it dies or never answers — so a stuck/broken server is
          // obvious in the terminal, not a silent hang.
          const deadline = Date.now() + 90000;
          let lastLog = 0;
          while (Date.now() < deadline) {
            if (exited) {
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
          try { process.kill(-proc.pid, 'SIGKILL'); } catch (e) { /* gone */ }
          throw new Error(
            `[e2e] ariadne serve NOT ready after 90s on http://${HOST}:${port}.\n`
            + `--- server output (tail) ---\n${out || '(no output — is `uv run ariadne serve` runnable from the repo root?)'}`);
        },

        async stopAriadne() {
          if (proc && proc.pid) {
            try { process.kill(-proc.pid, 'SIGTERM'); }
            catch (e) { try { proc.kill('SIGTERM'); } catch (_) { /* gone */ } }
            proc = null;
          }
          if (workspace) {
            try { fs.rmSync(workspace, { recursive: true, force: true }); } catch (e) { /* best effort */ }
            workspace = null;
          }
          return null;
        },
      });
    },
  },
});
