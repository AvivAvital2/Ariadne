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

async function waitFor(fn, want, ms) {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    if ((await fn()) === want) return true;
    await new Promise((r) => setTimeout(r, 300));
  }
  return (await fn()) === want;
}

module.exports = defineConfig({
  e2e: {
    // NOTE: no baseUrl on purpose. Cypress verifies a configured baseUrl at
    // launch (before any hook) — that would fail because each test boots its
    // own server. The support file sets baseUrl per test, after startup.
    specPattern: 'cypress/e2e/**/*.cy.js',
    supportFile: 'cypress/support/e2e.js',
    fixturesFolder: false,
    video: false,
    taskTimeout: 60000,
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

          proc = spawn('uv', ['run', 'ariadne', 'serve', '--host', HOST, '--port', String(port)], {
            cwd: REPO_ROOT,
            env: { ...process.env, ARIADNE_CONFIG: path.join(workspace, 'ariadne.yaml') },
            stdio: 'ignore',   // flip to 'inherit' to debug a startup failure
            detached: true,    // own process group → clean group kill (incl. the MCP child)
          });

          const up = await waitFor(() => ping(port), true, 45000);
          if (!up) {
            try { process.kill(-proc.pid, 'SIGKILL'); } catch (e) { /* already gone */ }
            throw new Error(`ariadne serve did not become ready on http://${HOST}:${port} (try stdio:'inherit')`);
          }
          return { baseUrl: `http://${HOST}:${port}`, sourcePath };
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
