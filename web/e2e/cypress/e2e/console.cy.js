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

// Exact diagram source from the reported console screenshot.
const POLICY_DIAGRAM = `sequenceDiagram
autonumber
participant AF as Authored fixture
participant EP as EXPIRY_POLICY constant
participant App as app
participant Unrelated as unrelated

AF->>EP: defines EXPIRY_POLICY
Note right of EP: token consumption removes a token before checking its deadline
AF->>App: documents shared source policy used by app as an authored dependency control
Note over AF,Unrelated: unrelated implements a separate retention policy and is outside this dependency scope`;

// Every /api is stubbed, so this spec only needs the page SERVED — it visits the
// PERSISTENT static server (booted once in cypress.config.js, fixed URL) rather
// than booting `ariadne serve` per spec. That per-spec boot caused the old
// `cypress run` boot-loop: on a spec re-attempt the server re-booted on a new
// port, so cy.visit chased a moving origin and Cypress's CDP context never
// settled (-32000 "Cannot find context"). A fixed URL recovers cleanly, like any
// external visit. The real server chain is covered by backend-atomicity.cy.js.
describe('Ask console', () => {
  beforeEach(() => {
    cy.intercept('POST', '/api/sources', { statusCode: 200, body: { default_source: 'proj', sources: [{ name: 'proj' }] } }).as('sources');
    cy.intercept('POST', '/api/ask', { statusCode: 200, body: ASK }).as('ask');
    cy.intercept('POST', '/api/search', { statusCode: 200, body: SEARCH }).as('search');
    cy.intercept('POST', '/api/feedback', { statusCode: 200, body: { success: true, message: 'Hit logged.' } }).as('feedback');
    cy.task('fixedStaticUrl').then((base) => cy.visit(base + '/static/console.html'));
  });

  it('loads with the source pill and default chips', () => {
    cy.wait('@sources');
    cy.get('#srcName').should('have.text', 'proj');
    cy.get('#chips .chip').should('have.length.greaterThan', 0);
  });

  it('ask: posts ask + search, renders answer, ranked rail, chips', () => {
    cy.get('#q').type('how are retries handled?');
    cy.get('#askBtn').click();
    cy.wait('@ask').its('request.body').should('deep.equal', { question: 'how are retries handled?', role: 'developer', source: 'proj' });
    cy.wait('@search').its('request.body').should('deep.equal', { query: 'how are retries handled?', role: 'developer', source: 'proj' });
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

  it('renders the reported Mermaid sequence as a diagram and preserves its source', () => {
    cy.viewport(1440, 1200);
    cy.intercept('POST', '/api/ask', {body: {...ASK,
      answer: '**Policy overview**\n\n```mermaid\n' + POLICY_DIAGRAM + '\n```\n\nEnd of explanation.'}}).as('diagramAsk');
    cy.get('#q').type('Generate a sequence diagram explaining how the policy is authored{enter}');
    cy.wait('@diagramAsk');
    cy.get('#answerCard .answer-diagram svg').should('be.visible')
      .and('contain', 'Authored fixture').and('contain', 'EXPIRY_POLICY constant')
      .and('contain', 'defines EXPIRY_POLICY');
    cy.viewport(480, 1000);
    cy.get('#answerCard .answer-diagram svg').should($svg => {
      const svg = $svg[0];
      expect(svg.getBoundingClientRect().width).to.be.at.least(svg.viewBox.baseVal.width - 1);
    });
    cy.get('#answerCard .answer-diagram').should($diagram => {
      expect($diagram[0].getBoundingClientRect().right)
        .to.be.at.most($diagram[0].closest('#answerCard').getBoundingClientRect().right);
    });
    cy.viewport(1440, 1200);
    cy.screenshot('policy-sequence-rendered', {capture: 'viewport'});
    cy.get('#answerCard b').should('contain', 'Policy overview');
    cy.get('#answerCard').should('contain', 'End of explanation.');
    cy.get('#answerCard details summary').contains('Diagram source').click();
    cy.get('#answerCard details pre code').should('have.text', POLICY_DIAGRAM);
    cy.get('#answerCard .fb button').contains('Helpful').click();
    cy.wait('@feedback').its('request.body.event_id').should('equal', 42);
  });

  it('keeps malformed diagrams and ordinary code readable while rendering later diagrams', () => {
    const invalid = 'sequenceDiagram\nthis is not a valid sequence';
    const code = 'if value < 2:\n    print("**literal** `<tag>`")\n```';
    cy.intercept('POST', '/api/ask', {body: {...ASK, answer:
      '**Before**\n\n```mermaid\n' + invalid + '\n```\n\n' +
      '````python\n' + code + '\n````\n\n' +
      '```mermaid\nflowchart LR\nA[First] --> B[Second]\n```\n\nAfter.'}});
    cy.get('#q').type('mixed diagrams{enter}');
    cy.get('#answerCard .diagram-error').should('be.visible').and('contain', 'could not be rendered');
    cy.get('#answerCard details').first().should('have.prop', 'open', true)
      .find('code').should('have.text', invalid);
    cy.get('#answerCard .answer-content > pre code').should('have.text', code)
      .find('b, tag').should('not.exist');
    cy.get('#answerCard .answer-diagram svg').should('have.length', 1).and('contain', 'Second');
    cy.get('#answerCard').should('contain', 'Before').and('contain', 'After.');
    cy.get('#status.err').should('not.exist');
    cy.get('#askBtn').should('be.enabled');
  });

  it('falls back when the diagram asset fails and retries rendering on the next answer', () => {
    const diagram = 'flowchart LR\nA[First] --> B[Second]';
    cy.intercept('POST', '/api/ask', {body: {...ASK, answer: '```mermaid\n' + diagram + '\n```'}});
    cy.intercept('GET', '/static/vendor/mermaid-12.2.0/mermaid.min.js', {statusCode: 503, body: ''});
    cy.get('#q').type('draw{enter}');
    cy.get('#answerCard .diagram-error').should('be.visible');
    cy.get('#answerCard details').should('have.prop', 'open', true).find('code').should('have.text', diagram);
    cy.get('#askBtn').should('be.enabled');
    cy.intercept('GET', '/static/vendor/mermaid-12.2.0/mermaid.min.js', req => req.continue());
    cy.get('#askBtn').click();
    cy.get('#answerCard .answer-diagram svg').should('be.visible').and('contain', 'Second');
    cy.get('#answerCard .diagram-error').should('not.exist');
  });

  it('escapes model HTML and refuses diagram configuration overrides without external assets', () => {
    const external = [];
    cy.intercept({url: /^https?:\/\/(?!localhost[:/]|127\.0\.0\.1[:/]).+/}, req => {
      external.push(req.url); req.destroy();
    });
    const unsafe = '<img src=x onerror="window.__diagramXss=true">';
    const directive = '%%{init: {"securityLevel":"loose","themeCSS":"body {display:none}"}}%%\nflowchart LR\nA-->B';
    const frontmatter = '---\nconfig:\n  securityLevel: loose\n---\nflowchart LR\nA-->B';
    const remoteImage = 'flowchart LR\nA@{ img: "https://example.invalid/probe.png", label: "Remote", h: 60 }';
    const diagram = 'flowchart LR\nA["<img src=x onerror=window.__diagramXss=true>"] --> B[Safe]\nclick B "javascript:window.__diagramXss=true"';
    cy.intercept('POST', '/api/ask', {body: {...ASK, answer:
      unsafe + '\n\n' + [diagram, directive, frontmatter, remoteImage].map(s => '```mermaid\n' + s + '\n```').join('\n\n')}});
    cy.get('#q').type('untrusted labels{enter}');
    cy.get('#answerCard .diagram-error').should('have.length', 3);
    cy.get('#answerCard .answer-diagram svg').should('have.length', 1).and('contain', 'Safe');
    cy.get('#answerCard img, #answerCard script, #answerCard a[href^="javascript:"]').should('not.exist');
    cy.get('#answerCard .answer-content > p').should('contain', unsafe);
    cy.get('#answerCard details').eq(1).should('have.prop', 'open', true).find('code').should('have.text', directive);
    cy.get('#answerCard details').eq(2).find('code').should('have.text', frontmatter);
    cy.get('#answerCard details').eq(3).find('code').should('have.text', remoteImage);
    cy.window().then(win => { expect(win.__diagramXss).to.equal(undefined); expect(external).to.deep.equal([]); });
    cy.get('#askBtn').should('be.enabled');
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
