/// <reference types="cypress" />
/*
 * E2E for the Ask console (web/static/console.html).
 *
 * All /api/* calls are stubbed with cy.intercept, so this exercises the real
 * page in a real browser without needing an LLM, embeddings, API keys, or an
 * onboarded source — only `ariadne serve` running (to serve the static page).
 * Canned shapes match ariadne_mcp/models.py (AskResponse / SearchResponse).
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

describe('Ask console', () => {
  beforeEach(() => {
    cy.intercept('POST', '/api/sources', { statusCode: 200, body: { default_source: 'proj', sources: [{ name: 'proj' }] } }).as('sources');
    cy.intercept('POST', '/api/ask', { statusCode: 200, body: ASK }).as('ask');
    cy.intercept('POST', '/api/search', { statusCode: 200, body: SEARCH }).as('search');
    cy.intercept('POST', '/api/feedback', { statusCode: 200, body: { success: true, message: 'Hit logged.' } }).as('feedback');
    cy.visit('/static/console.html');
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
    cy.get('#rail').should('contain', 'The @retryable decorator')
      .and('contain', '0.94').and('contain', 'http/resilience.py');
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
