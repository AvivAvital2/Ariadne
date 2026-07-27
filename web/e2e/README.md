# Ariadne web UI — Cypress E2E

Real-browser end-to-end tests for the Ask console and onboarding wizard.

**The tests start (and stop) Ariadne themselves — one fresh, isolated server
per test.** Before each test, a task boots `ariadne serve` with a brand-new temp
config dir (empty DB + empty cache) on a unique free port; after the test it's
killed and the temp dir removed. So every test is **atomic** — no cache or
residual data carries between tests. You do **not** start a server manually.

## Prerequisites
- `uv` + this repo (the harness runs `uv run ariadne serve` from the repo root).
- Node (for Cypress).

## Run
```bash
cd web/e2e
npm install
npx cypress run        # headless   (or: npx cypress open)
```

## Specs
- `cypress/e2e/console.cy.js` — Ask console (stubs `/api/*`): load, ask →
  answer + confidence + ranked-sources rail + chips, feedback, role toggle,
  error handling.
- `cypress/e2e/onboarding.cy.js` — wizard (stubs `/api/*`): shell load, and the
  full build Connect → Start build → **SSE progress** → ready → console handoff.
- `cypress/e2e/backend-atomicity.cy.js` — **real backend, no stubs, no LLM**:
  drives the live per-test server via `cy.request` (source_add / list / discover
  / estimate) and proves each test starts clean (no leaked state).

## How the harness works
- `cypress.config.js` → `setupNodeEvents` defines `startAriadne` / `stopAriadne`
  tasks (spawn on a free port with a temp `ARIADNE_CONFIG`; group-kill on stop).
- `cypress/support/e2e.js` → `beforeEach` starts a server + sets `baseUrl`;
  `afterEach` stops it.
- No `baseUrl` in the config on purpose: Cypress verifies a configured baseUrl at
  launch (before any hook), which would fail against per-test startup.

## Notes / knobs
- **Per-test restart cost:** each test pays one `ariadne serve` boot (a few
  seconds, incl. the stdio MCP child). If that's too slow, move the hooks in
  `support/e2e.js` from `beforeEach`/`afterEach` to `before`/`after` for
  per-*spec* freshness instead of per-*test*.
- **Debugging startup:** flip `stdio: 'ignore'` → `'inherit'` in
  `cypress.config.js` to see `ariadne serve`'s output if it won't come up.
- **SSE spec caveat:** `onboarding.cy.js` stubs the `/api/onboard/events` stream
  with a static `text/event-stream` body; if your Cypress build doesn't drive
  `EventSource` from a static intercept, that one flow may need a streaming stub.
- A DOM-level equivalent that runs with no browser/socket lives at
  `tests/test_console_ui.js` (jsdom + `node --test`).
