// Server lifecycle is chosen per spec (not globally), because freshness only
// matters where server state is actually touched:
//   • console.cy.js / onboarding.cy.js — stub every /api/*, so the server only
//     serves static pages and its state never matters → ONE fresh server per
//     spec (before/after). Booting a fresh server per test there is pure cost.
//   • backend-atomicity.cy.js — hits the REAL backend → a fresh server per
//     TEST (beforeEach/afterEach) for genuine atomicity.
// Each spec sets Cypress.env('base') from the startAriadne task and uses
// absolute URLs, so there is no runtime-baseUrl dependency.
