/// <reference types="cypress" />
/*
 * E2E for the onboarding wizard (web/static/onboarding.html) — served at `/`.
 * Only `ariadne serve` is needed (the pages are static). The full 5-step build
 * flow (which streams progress over SSE) is a deeper spec; here we cover the
 * shell load and the Phase-3 handoff into the Ask console.
 */
describe('Onboarding wizard', () => {
  it('loads the wizard shell', () => {
    cy.visit('/');
    cy.get('#nextBtn').should('be.visible'); // the Continue button on step 1
  });

  it('ready screen is wired to hand off into the Ask console (Phase 3)', () => {
    cy.visit('/');
    // The ready panel is hidden until a build completes; assert the handoff
    // link is wired (exists, correct target + label). Driving a full stubbed
    // build (incl. the SSE progress stream) to click it live is a deeper spec.
    cy.get('#gen-ready a[href="/static/console.html"]')
      .should('exist')
      .and('contain', 'Ask');
  });
});
