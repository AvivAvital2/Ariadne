/// <reference types="cypress" />
/*
 * Real-backend E2E — NO stubs, NO UI, NO LLM. A FRESH server PER TEST (see the
 * beforeEach below) drives the live per-test Ariadne via cy.request to prove:
 *   (1) the no-LLM MCP tool chain works over real HTTP (source_add / list /
 *       discover / estimate), and
 *   (2) each test gets a clean server — state never leaks between tests.
 */
describe('live backend — fresh + atomic per test', () => {
  beforeEach(() => cy.task('startAriadne').then((s) => {
    Cypress.env('base', s.baseUrl);
    Cypress.env('sourcePath', s.sourcePath);
  }));
  afterEach(() => cy.task('stopAriadne'));

  const api = (p) => Cypress.env('base') + p;

  it('a fresh server starts with zero sources', () => {
    cy.request('POST', api('/api/sources'), {}).its('body.sources').should('have.length', 0);
  });

  it('source_add registers a source on THIS server; discover + estimate then work', () => {
    const path = Cypress.env('sourcePath');

    cy.request('POST', api('/api/source_add'), { name: 'projA', path }).its('body').should((b) => {
      expect(b.source).to.eq('projA');
      expect(b.created).to.eq(true);
    });

    cy.request('POST', api('/api/sources'), {}).its('body.sources').should((s) =>
      expect(s.map((x) => x.name)).to.include('projA'));

    cy.request('POST', api('/api/discover'), { source: 'projA' }).its('body').should((d) => {
      expect(d.manifest_written).to.eq(true);
      expect(d.languages.map((l) => l.language)).to.include('python');
    });

    cy.request('POST', api('/api/estimate'), { source: 'projA' })
      .its('body.file_count').should('be.greaterThan', 0);
  });

  it("the previous test's source did NOT leak — this server is clean", () => {
    cy.request('POST', api('/api/sources'), {}).its('body.sources').should('have.length', 0);
  });
});
