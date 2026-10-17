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

The Qwen Docker journey builds the current application image, runs its browser
and MCP services, and uses two deterministic HTTP model fixtures. It requires
Docker Desktop (or Docker Engine), but no GPU, model weights or API keys:

```bash
CYPRESS_CONTAINER=qwen npx cypress run --spec cypress/e2e/container-journey.cy.js
```

It checks the configured model, an offline cost preview with zero provider
charges and no database writes, real Python/Go indexing and onboarding,
semantic search, answers, clew selection and source evidence, and persistence
after restarting the application. Browser requests use the configured Qwen
pair without model or source overrides from the test. These are application
integration checks; they do not qualify Qwen weights, vLLM or GPU performance.

`CYPRESS_CONTAINER=openai` and `CYPRESS_CONTAINER=anthropic` run the same
container journey against paid cloud APIs. `./run-container-suite.sh` runs all
three variants; `--parallel` gives each its own ports, Compose project and state.

`hosted-onboarding.cy.js` is an additional paid smoke for an already-running
application with all three cloud credentials. Mount a one-file synthetic Go
retry project at `/workspace/retry-demo`, then run:

```bash
CYPRESS_HOSTED_ONBOARDING_URL=http://127.0.0.1:8765 npx cypress run --spec cypress/e2e/hosted-onboarding.cy.js
```

It chooses hosted Qwen in Preview, refuses a preview with
more than one file, generates an explanation, and checks search and Ask after
reloading the saved project. It does not stub model requests or tool responses.

For a free comparison in the same Docker application, set
`CYPRESS_PREVIEW_ONBOARDING_URL=http://127.0.0.1:8765` instead of
`CYPRESS_HOSTED_ONBOARDING_URL`. This runs the actual GPT, Claude and hosted Qwen
cost previews and checks that no project was saved. It never starts a build.

## Specs
- `cypress/e2e/onboarding-models.cy.js` — compares all four pairs in Preview, checks paid/zero cost switching, saves the final choice only on Start build, and checks console reload without a model override on Ask.
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
