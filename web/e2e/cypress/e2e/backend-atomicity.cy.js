/// <reference types="cypress" />
/*
 * Real-backend E2E — NO stubs, NO UI, NO LLM. Drives the live per-test server
 * directly via cy.request to prove two things:
 *   (1) the no-LLM MCP tool chain works over real HTTP (source_add / list /
 *       discover / estimate), and
 *   (2) each test gets a clean server — state from one test never leaks into
 *       the next (the whole point of the per-test fresh-server harness).
 *
 * The console/onboarding specs stub /api/*; this spec deliberately does not.
 */
describe('live backend — fresh + atomic per test', () => {
  it('a fresh server starts with zero sources', () => {
    cy.request('POST', '/api/sources', {}).its('body.sources').should('have.length', 0);
  });

  it('source_add registers a source on THIS server; discover + estimate then work', () => {
    const path = Cypress.env('sourcePath');

    cy.request('POST', '/api/source_add', { name: 'projA', path }).its('body').should((b) => {
      expect(b.source).to.eq('projA');
      expect(b.created).to.eq(true);
    });

    cy.request('POST', '/api/sources', {}).its('body.sources').should((s) =>
      expect(s.map((x) => x.name)).to.include('projA'));

    cy.request('POST', '/api/discover', { source: 'projA' }).its('body').should((d) => {
      expect(d.manifest_written).to.eq(true);
      expect(d.languages.map((l) => l.language)).to.include('python');
    });

    cy.request('POST', '/api/estimate', { source: 'projA' })
      .its('body.file_count').should('be.greaterThan', 0);
  });

  it("the previous test's source did NOT leak — this server is clean", () => {
    // Atomicity proof: projA added above is gone, because this is a brand-new
    // server with an empty config + cache.
    cy.request('POST', '/api/sources', {}).its('body.sources').should('have.length', 0);
  });
});
