/// <reference types="cypress" />
// FULL REAL JOURNEY — the end-to-end proof: source → onboarding wizard →
// (mocked build) → REAL generate → Ask console → REAL question → REAL answer.
//
// This is a SLOW, REAL test: it boots a real `ariadne serve`, runs a real
// `ariadne generate`, and does a real `ariadne_ask`. It therefore REQUIRES the
// LLM + embedding API keys in the run env (ANTHROPIC_API_KEY + the OpenAI key)
// and takes minutes. It's excluded from the default suite — run it explicitly:
//
//   cd web/e2e && npx cypress run --spec cypress/e2e/journey.cy.js
//
// The wizard's real /api calls (source_add / discover / estimate / ask) are
// intercepted only to alias+wait on them (no stub body → they pass through to
// the real server). Only the expensive build SSE is stubbed; the real docs come
// from the `journeyGenerate` task. The server uses a FIXED URL (persistent,
// 127.0.0.1) so cy.visit survives Cypress's re-attempt — same fix as the other
// UI specs.
// Gated so the default suite stays fast + key-free. Enable explicitly:
//   cd web/e2e && CYPRESS_JOURNEY=1 npx cypress run --spec cypress/e2e/journey.cy.js
const RUN_JOURNEY = String(Cypress.env('JOURNEY')) === '1' || Cypress.env('JOURNEY') === true;

(RUN_JOURNEY ? describe : describe.skip)('Journey: onboard → generate → ask (real answer)', () => {
  let server;   // { baseUrl, sourceName, sourcePath, configPath }
  before(() => cy.task('journeyServer').then((s) => { server = s; }));

  it('onboards a source, generates real docs, and answers a real question about them', () => {
    // Alias the REAL calls (no stub body = pass-through); stub ONLY the build.
    cy.intercept('POST', '/api/source_add').as('sourceAdd');
    cy.intercept('POST', '/api/discover').as('discover');
    cy.intercept('POST', '/api/estimate').as('estimate');
    cy.intercept('POST', '/api/onboard', { statusCode: 200, body: { job_id: 'j1', status: 'running' } }).as('onboard');
    cy.intercept('GET', '/api/onboard/events*', {
      statusCode: 200,
      headers: { 'content-type': 'text/event-stream' },
      body:
        'data: {"type":"progress","current":1,"total":1,"message":"Generating documentation"}\n\n' +
        'data: {"type":"done","result":{"files_indexed":1,"docs_written":1,"themes_found":0,"coverage_percent":100}}\n\n',
    }).as('events');

    // 1 — Onboarding wizard, driving the REAL backend.
    cy.visit(server.baseUrl + '/');
    cy.get('#srcName').type(server.sourceName);
    cy.get('#srcPath').type(server.sourcePath);
    cy.get('#nextBtn').click();                                          // Connect → step 2
    cy.wait('@sourceAdd').its('response.statusCode').should('eq', 200);
    cy.wait('@discover', { timeout: 60000 }).its('response.statusCode').should('eq', 200);  // step 2 onEnter
    cy.get('#nextBtn').click();                                          // → step 3 (Preview)
    cy.wait('@estimate', { timeout: 60000 }).its('response.statusCode').should('eq', 200);
    cy.get('#nextBtn').click();                                          // → step 4
    cy.get('#nextBtn').click();                                          // → step 5 (Summary)

    // 2 — Mocked build → ready screen (fast; real docs come next).
    cy.get('#nextBtn').should('contain', 'Start build').click();
    cy.wait('@onboard');
    cy.get('#gen-ready', { timeout: 20000 }).should('be.visible');

    // 3 — REAL generate so the KB has genuine, embedded docs about retryutils.
    cy.task(
      'journeyGenerate',
      { sourceName: server.sourceName, configPath: server.configPath },
      { timeout: 600000 },
    ).then((r) => {
      cy.task('log', `[journey] generate rc=${r.rc}`);
      expect(r.rc, `ariadne generate must succeed. tail:\n${r.output}`).to.eq(0);
    });

    // 4 — Hand off to the Ask console (same real server).
    cy.get('#gen-ready a[href="/static/console.html"]').should('be.visible').click();
    cy.location('pathname').should('eq', '/static/console.html');

    // 5 — Ask a REAL question → REAL, non-stubbed answer that reflects the code.
    // FINDING: the Ask console posts to BOTH /api/ask {question,role} and
    // /api/search {query,role} with NO source, and hides the answer if EITHER
    // throws (postJSON throws on non-200). Both are source-scoped, and here the
    // server's cwd isn't inside the source tree, so we scope BOTH to the
    // onboarded source at the request layer. (If the console sent its selected
    // source, this wouldn't be needed.)
    const scopeToSource = (req) => { req.body = { ...req.body, source: server.sourceName }; };
    cy.intercept('POST', '/api/ask', scopeToSource).as('ask');
    cy.intercept('POST', '/api/search', scopeToSource).as('search');
    cy.get('#q').type('how are retries handled in this project?{enter}');
    cy.wait('@ask', { timeout: 120000 }).then((i) => {
      // On failure the assertion message carries the real {error: "..."}; on
      // success we don't dump the whole answer to the terminal.
      expect(i.response.statusCode, `ask failed: ${JSON.stringify(i.response.body)}`).to.eq(200);
    });
    cy.get('#answerCard', { timeout: 120000 }).should('be.visible');
    // A genuine answer about the retry module — it should mention retry/backoff.
    cy.get('#answerCard').invoke('text').should('match', /retr|backoff/i);
  });
});
