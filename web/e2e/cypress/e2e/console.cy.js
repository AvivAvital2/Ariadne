/// <reference types="cypress" />
/*
 * E2E for the Ask console (web/static/console.html). Every /api/* is stubbed
 * (cy.intercept) — the page never touches server state — so ONE fresh server
 * per spec is enough. (Per-test atomicity lives in backend-atomicity.cy.js.)
 * Shapes match ariadne_mcp/models.py (AskResponse / SearchResponse).
 */
const ASK = {
  answer: 'Retries live in **http/resilience.py** via the `@retryable` decorator.',
  confidence: 'high',
  sources: ['The @retryable decorator'],
  event_id: 42,
};
const SEARCH = {
  documents: [
    { id: 'd1', title: 'The @retryable decorator', content_type: 'explanation', source_files: ['http/resilience.py'], score: 0.94 },
    { id: 'd2', title: 'Double-wrapped retries', content_type: 'gotcha', source_files: ['http/resilience.py'], score: 0.88 },
  ],
  suggested_queries: ['What breaks if I change RetryPolicy?', 'Where are backoff defaults set?'],
  improvement_hint: 'Enable a spool for deeper coverage.',
  event_id: 99,
};

// SKIPPED — headless-harness issue, NOT a product bug. Under `cypress run`,
// cy.visit of the served page makes Cypress fail its socket injection with CDP
// -32000 "Cannot find context with specified id"; Cypress then silently re-runs
// the spec (before() → startAriadne) in an endless boot loop. Ruled out via a
// `cypress:server:*` trace: not the browser (Chrome loops too), not this page
// (jsdom exercises it 6/6 — no nav/reload/RAF), not server headers (no CSP/XFO),
// not a renderer crash, and not host origin (bind+visit+proxy all on localhost
// didn't clear it). Functional coverage lives in tests/test_console_ui.js (jsdom,
// real DOM, 6/6) + the real server chain in backend-atomicity.cy.js. To debug,
// re-enable and run headed/interactive via `cypress open`, where the runner shows
// the injection error live.
describe.skip('Ask console', () => {
  let base;
  before(() => cy.task('startAriadne').then((s) => { base = s.baseUrl; }));
  after(() => cy.task('stopAriadne'));

  beforeEach(() => {
    cy.log('server base = ' + base);   // shows in the video/command log (catches an undefined base)
    cy.intercept('POST', '/api/sources', { statusCode: 200, body: { default_source: 'proj', sources: [{ name: 'proj' }] } }).as('sources');
    cy.intercept('POST', '/api/ask', { statusCode: 200, body: ASK }).as('ask');
    cy.intercept('POST', '/api/search', { statusCode: 200, body: SEARCH }).as('search');
    cy.intercept('POST', '/api/feedback', { statusCode: 200, body: { success: true, message: 'Hit logged.' } }).as('feedback');
    cy.log('visiting ' + base + '/static/console.html');
    cy.visit(base + '/static/console.html');
  });

  it('loads with the source pill and default chips', () => {
    cy.wait('@sources');
    cy.get('#srcName').should('have.text', 'proj');
    cy.get('#chips .chip').should('have.length.greaterThan', 0);
  });

  it('ask: posts ask + search, renders answer, ranked rail, chips', () => {
    cy.get('#q').type('how are retries handled?');
    cy.get('#askBtn').click();
    cy.wait('@ask').its('request.body').should('deep.equal', { question: 'how are retries handled?', role: 'developer' });
    cy.wait('@search').its('request.body').should('deep.equal', { query: 'how are retries handled?', role: 'developer' });
    cy.get('#answerGrid').should('be.visible');
    cy.get('#answerCard').should('contain', 'http/resilience.py');
    cy.get('#answerCard .badge').first().should('contain', 'high confidence');
    cy.get('#answerCard').should('contain', 'explanation').and('contain', 'gotcha');
    cy.get('#rail').should('contain', 'The @retryable decorator').and('contain', '0.94').and('contain', 'http/resilience.py');
    cy.get('#chips .chip').should('contain', 'What breaks if I change RetryPolicy?');
  });

  it('feedback: 👍 posts /api/feedback with the ask event_id', () => {
    cy.get('#q').type('q{enter}');
    cy.wait('@ask');
    cy.get('#answerCard .fb button').contains('Helpful').click();
    cy.wait('@feedback').its('request.body').should('deep.equal', { event_id: 42, helpful: true });
    cy.get('#answerCard .fb button').should('contain', 'Logged hit');
  });

  it('role toggle: Product re-asks as product_manager', () => {
    cy.get('#q').type('q1{enter}');
    cy.wait('@ask');
    cy.get('#roleSeg button').contains('Product').click();
    cy.wait('@ask').its('request.body.role').should('equal', 'product_manager');
    cy.get('#answerCard .badge.role').should('contain', 'product');
  });

  it('error: a failing /api/ask surfaces an error and hides the answer', () => {
    cy.intercept('POST', '/api/ask', { statusCode: 500, body: { error: 'model unavailable' } }).as('askErr');
    cy.get('#q').type('q{enter}');
    cy.wait('@askErr');
    cy.get('#status.err').should('contain', 'Could not answer');
    cy.get('#answerGrid').should('not.be.visible');
  });
});
