/// <reference types="cypress" />
// FULL CONTAINER JOURNEY — the automated Phase-0 acceptance the blueprint
// (designs/web-ui/console-and-deployment.md §5) left MANUAL. Boots the REAL image
// via `docker compose` (the actual deploy path) and drives:
//   source → onboarding wizard → REAL ariadne_onboard INSIDE the container
//   (discover → scip-python + scip-go index → REAL `scip merge` → catalog →
//   generate → embed → themes) → Ask console → REAL question → REAL answer.
//
// TWO variants, both important (embeddings are ALWAYS OpenAI — Anthropic has no
// embeddings API — so the provider only selects the GENERATION backend):
//   • openai    — gpt generation + OpenAI embeddings   → one key  (OPENAI_API_KEY)
//   • anthropic — Claude generation + OpenAI embeddings → both keys
// The mounted source is multi-language (Python + Go) so the real `scip merge`
// runs in BOTH variants (single-language would skip merge — cli/index.py:798).
//
// SLOW + REAL: builds the image + runs a real onboard (LLM + embeddings), so it
// costs a little. Gated — run BOTH: CYPRESS_CONTAINER=1 ; or ONE:
// CYPRESS_CONTAINER=openai / =anthropic. Needs a reachable Docker daemon + the
// relevant API key(s).
//   cd web/e2e && CYPRESS_CONTAINER=1 npx cypress run --spec cypress/e2e/container-journey.cy.js
const V = String(Cypress.env('CONTAINER') ?? '');
const RUN_ALL = V === '1' || V === 'true' || V === true;

// Each variant gets its own compose project, host ports, and image tag so the
// suite runner can launch both SIMULTANEOUSLY without collisions (see
// web/e2e/run-container-suite.sh).
const VARIANTS = [
  { key: 'openai', title: 'all-OpenAI (gpt generation + OpenAI embeddings)', model: 'gpt-5.5', requireAnthropic: false, project: 'ariadne-e2e-openai', webPort: 18765, mcpPort: 18000 },
  { key: 'anthropic', title: 'cross-provider (Claude generation + OpenAI embeddings)', model: 'claude-opus-4-8', requireAnthropic: true, project: 'ariadne-e2e-anthropic', webPort: 18775, mcpPort: 18010 },
];

VARIANTS.forEach((variant) => {
  const run = RUN_ALL || V === variant.key;
  (run ? describe : describe.skip)(`Container journey — ${variant.title}`, () => {
    let server;   // { baseUrl, sourceName, sourcePath, workspace }
    // Build (if needed) + boot the stack — allow up to 30 min for a cold build.
    before(() => cy.task('containerUp', {
      requireAnthropic: variant.requireAnthropic,
      project: variant.project,
      webPort: variant.webPort,
      mcpPort: variant.mcpPort,
      variantKey: variant.key,
    }, { timeout: 1800000 }).then((s) => { server = s; }));
    after(() => cy.task('containerDown'));

    it('onboards a mounted multi-language source and answers a real question about it', () => {
      // Force THIS variant's generation model into estimate + onboard, so the
      // provider resolves to the matching backend (gpt-* → openai, claude-* →
      // anthropic) regardless of the wizard's default picker. Nothing else is
      // stubbed — the container runs the real onboard + ask.
      const setModel = (req) => { req.body = { ...req.body, model: variant.model }; };
      cy.intercept('POST', '/api/source_add').as('sourceAdd');
      cy.intercept('POST', '/api/discover').as('discover');
      cy.intercept('POST', '/api/estimate', setModel).as('estimate');
      cy.intercept('POST', '/api/onboard', setModel).as('onboard');
      cy.intercept('GET', '/api/onboard/events*').as('events');

      // 1 — Onboarding wizard against the real containerized backend.
      cy.visit(server.baseUrl + '/');
      cy.get('#srcName').type(server.sourceName);
      cy.get('#srcPath').type(server.sourcePath);                 // /workspace/retryutils (the mount)
      cy.get('#nextBtn').click();                                 // Connect → step 2
      cy.wait('@sourceAdd').its('response.statusCode').should('eq', 200);
      cy.wait('@discover', { timeout: 120000 }).its('response.statusCode').should('eq', 200);  // step 2 onEnter
      cy.get('#nextBtn').click();                                 // → step 3 (Preview)
      cy.wait('@estimate', { timeout: 120000 }).its('response.statusCode').should('eq', 200);
      cy.get('#nextBtn').click();                                 // → step 4
      cy.get('#nextBtn').click();                                 // → step 5 (Summary)

      // 2 — REAL build INSIDE the container: index (python + go) → `scip merge` →
      //     catalog → generate → embed → themes. Slow — wait for the ready screen.
      //     Backstop only; the streamed [container-log] shows live progress.
      cy.get('#nextBtn').should('contain', 'Start build').click();
      cy.wait('@onboard').its('response.statusCode').should('eq', 200);
      cy.get('#gen-ready', { timeout: 600000 }).should('be.visible');   // real onboard: up to 10m

      // 3 — Hand off to the Ask console (same container).
      cy.get('#gen-ready a[href="/static/console.html"]').should('be.visible').click();
      cy.location('pathname').should('eq', '/static/console.html');

      // 4 — REAL question → REAL answer. The console posts no source and hides the
      //     answer if /api/search errors, so scope BOTH ask + search to the
      //     onboarded source at the request layer (same finding as journey.cy.js).
      const scopeToSource = (req) => { req.body = { ...req.body, source: server.sourceName }; };
      cy.intercept('POST', '/api/ask', scopeToSource).as('ask');
      cy.intercept('POST', '/api/search', scopeToSource).as('search');
      cy.get('#q').type('how are retries handled in this project?{enter}');
      cy.wait('@ask', { timeout: 180000 }).then((i) => {
        expect(i.response.statusCode, `ask failed: ${JSON.stringify(i.response.body)}`).to.eq(200);
      });
      cy.get('#answerCard', { timeout: 180000 }).should('be.visible');
      cy.get('#answerCard').invoke('text').should('match', /retr|backoff/i);
    });
  });
});
