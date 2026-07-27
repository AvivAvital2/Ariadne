const { defineConfig } = require('cypress');

module.exports = defineConfig({
  e2e: {
    // The Ariadne web UI (ariadne serve). Console: /static/console.html; wizard: /
    baseUrl: 'http://127.0.0.1:8765',
    specPattern: 'cypress/e2e/**/*.cy.js',
    supportFile: false,
    fixturesFolder: false,
    video: false,
  },
});
