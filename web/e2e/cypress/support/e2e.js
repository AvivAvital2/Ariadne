// Server lifecycle is chosen per spec (not globally), because freshness only
// matters where server state is actually touched:
//   • console.cy.js / onboarding.cy.js — stub every /api/*, so the server only
//     serves static pages and its state never matters → ONE fresh server per
//     spec (before/after). Booting a fresh server per test there is pure cost.
//   • backend-atomicity.cy.js — hits the REAL backend → a fresh server per
//     TEST (beforeEach/afterEach) for genuine atomicity.
// Each spec sets the base URL (closure var) from the startAriadne task and uses
// absolute URLs, so there is no runtime-baseUrl dependency.

// Surface every test FAILURE in the terminal (not just the Cypress UI), so the
// exact assertion error is visible from a terminal-only view. `function` (not
// an arrow) so `this.currentTest` is available.
afterEach(function () {
  const t = this.currentTest;
  if (t && t.state === 'failed' && t.err) {
    const msg = String(t.err.message || 'unknown').split('\n').slice(0, 4).join('  |  ');
    cy.task('log', `[TEST FAIL] ${t.fullTitle()}  ::  ${msg}`, { log: false });
  }
});
