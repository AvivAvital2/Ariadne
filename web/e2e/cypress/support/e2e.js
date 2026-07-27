// Every test gets its own fresh, isolated Ariadne server — a new temp config
// dir (empty DB + empty cache) on a unique port — started before the test and
// torn down after. So tests are atomic: no cache or residual data carries over.
// The startAriadne task hands back the baseUrl + a synthetic source path.
beforeEach(() => {
  cy.task('startAriadne').then(({ baseUrl, sourcePath }) => {
    Cypress.config('baseUrl', baseUrl);
    Cypress.env('sourcePath', sourcePath);
  });
});

afterEach(() => {
  cy.task('stopAriadne');
});
