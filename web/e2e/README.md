# Ariadne web UI — Cypress E2E

Real-browser end-to-end tests for the Ask console and the onboarding wizard.
Every `/api/*` call is stubbed with `cy.intercept`, so these exercise the UI in
isolation — **no LLM, API keys, embeddings, or onboarded source required.** You
only need the static pages served.

## Run

1. Serve the pages (from the repo root):

   ```bash
   uv run ariadne serve        # serves on http://127.0.0.1:8765
   ```

2. Install + run Cypress (from this directory):

   ```bash
   cd web/e2e
   npm install
   npx cypress run             # headless   (or: npx cypress open)
   ```

## Specs

- `cypress/e2e/console.cy.js` — Ask console: initial load (source pill + chips),
  ask → answer card + confidence + ranked-sources rail (with scores) + chips,
  👍 feedback, Developer/Product role toggle, and error handling.
- `cypress/e2e/onboarding.cy.js` — wizard shell load + the Phase-3 handoff link
  into the console.

Config: `cypress.config.js` (`baseUrl: http://127.0.0.1:8765`).

> The stub shapes mirror `ariadne_mcp/models.py` (`AskResponse`,
> `SearchResponse`). A DOM-level equivalent that runs without a browser lives at
> `tests/test_console_ui.js` (jsdom + `node --test`).
