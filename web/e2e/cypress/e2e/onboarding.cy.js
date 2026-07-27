/// <reference types="cypress" />
/*
 * E2E for the onboarding wizard (web/static/onboarding.html, served at /).
 * /api/* stubbed → ONE fresh server per spec. Covers the shell load and the
 * full build (Connect → Start build → SSE progress → ready → console handoff).
 */
// Visits the PERSISTENT static server (see console.cy.js header) instead of
// booting `ariadne serve` per spec — the per-spec boot was the old cy.visit
// boot-loop (a spec re-attempt re-booted on a new port → moving origin → CDP
// -32000 "Cannot find context"). /api is fully stubbed, so only the served page
// is needed; a fixed URL recovers cleanly.
describe('Onboarding wizard', () => {
  let base;
  beforeEach(() => cy.task('fixedStaticUrl').then((u) => { base = u; }));

  const wizard = () => base + '/';

  it('loads the wizard shell', () => {
    cy.intercept('POST', '/api/sources', { statusCode: 200, body: { default_source: null, sources: [] } });
    cy.visit(wizard());
    cy.get('#nextBtn').should('be.visible');
  });

  it('ready screen is wired to hand off into the Ask console (Phase 3)', () => {
    cy.intercept('POST', '/api/sources', { statusCode: 200, body: { default_source: null, sources: [] } });
    cy.visit(wizard());
    cy.get('#gen-ready a[href="/static/console.html"]').should('exist').and('contain', 'Ask');
  });

  it('full build: Connect → Start build → SSE progress → ready → handoff to console', () => {
    cy.intercept('POST', '/api/sources', { statusCode: 200, body: { default_source: null, sources: [] } }).as('sources');
    cy.intercept('POST', '/api/source_add', {
      statusCode: 200,
      body: { source: 'proj', created: true, path: '/tmp/proj', is_default: true, depends_on: [], branches: [], exclude: [], exclude_dirs: [], exempt_dirs: [], doc_types_by_language: {} },
    }).as('sourceAdd');
    cy.intercept('POST', '/api/discover', {
      statusCode: 200,
      body: { source: 'proj', file_count: 2, dir_count: 1, languages: [{ language: 'python', files: 2, percent: 100 }], indexers: [], index_kinds: [], manifest_written: true },
    }).as('discover');
    cy.intercept('POST', '/api/estimate', {
      statusCode: 200,
      body: {
        source: 'proj', model: 'claude-opus-4-8', input_per_million: 5, output_per_million: 25,
        file_count: 2, total_calls: 2, input_tokens: 100, output_tokens: 50, embedding_tokens: 10,
        total_cost_usd: 0.5, total_cost_batched_usd: 0.25, cost_lower_bound: 0.4, cost_upper_bound: 0.75,
        embedding_cost_usd: 0.01, languages: [{ language: 'python', files: 2, percent: 100 }],
        by_doc_type: [], by_directory: [], available_models: [{ model: 'claude-opus-4-8', input_per_million: 5, output_per_million: 25 }],
        language_doc_types: {}, exclusion_savings: [],
      },
    }).as('estimate');
    cy.intercept('POST', '/api/onboard', { statusCode: 200, body: { job_id: 'job-123', status: 'running' } }).as('onboard');
    // SSE stream: two progress frames then a terminal done (with the ready stats).
    // (If EventSource doesn't consume a static intercept body in your Cypress
    // build, this is the one spec that may need a streaming-stub tweak.)
    cy.intercept('GET', '/api/onboard/events*', {
      statusCode: 200,
      headers: { 'content-type': 'text/event-stream' },
      body:
        'data: {"type":"progress","current":1,"total":3,"message":"Describing catalog elements"}\n\n' +
        'data: {"type":"progress","current":2,"total":3,"message":"Generating documentation"}\n\n' +
        'data: {"type":"done","result":{"files_indexed":2,"docs_written":7,"themes_found":3,"coverage_percent":80}}\n\n',
    }).as('events');

    cy.visit(wizard());

    // Step 1 — Connect: name + path → source_add
    cy.get('#srcName').type('proj');
    cy.get('#srcPath').type('/tmp/proj');
    cy.get('#nextBtn').click();
    cy.wait('@sourceAdd').its('request.body.name').should('eq', 'proj');
    cy.wait('@discover');                 // step 2 onEnter
    cy.get('#nextBtn').click();           // → step 3
    cy.wait('@estimate');                 // step 3 onEnter
    cy.get('#nextBtn').click();           // → step 4
    cy.get('#nextBtn').click();           // → step 5 (Summary)

    // Step 5 — Start build (the button re-binds to startBuild here)
    cy.get('#nextBtn').should('contain', 'Start build').click();
    cy.wait('@onboard').its('request.body.source').should('eq', 'proj');
    cy.wait('@events');

    // SSE 'done' → ready screen with the real stats + the console handoff
    cy.get('#gen-ready', { timeout: 10000 }).should('be.visible');
    cy.get('#readyMsg').should('contain', '7 docs').and('contain', '3 themes').and('contain', '80%');
    cy.get('#gen-ready a[href="/static/console.html"]').should('be.visible').click();
    cy.location('pathname').should('eq', '/static/console.html');
  });
});
