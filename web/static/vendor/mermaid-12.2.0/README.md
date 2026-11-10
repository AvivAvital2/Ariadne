# Bundled Mermaid 12.2.0

`mermaid.min.js` and `LICENSE` are unchanged files from the official npm package:
<https://registry.npmjs.org/mermaid/-/mermaid-12.2.0.tgz>.
The standalone browser bundle includes its dependencies and their license
notices; it requires no ESM chunks or CDN at runtime.

- Archive npm integrity: `sha512-w7nr/SFrLO7BaVTs3dQTvYSV6nYSBj0Yy9DjBhsIJtAcHqRzXSq4Pg32ujHZ/AdWsAMNEvvhbm5dt9E2tOiHzw==`
- Bundle SHA-256: `a8a6a6f805b074a8261a2825c86f99c8054acb3b35f50bb7e24a4f9f2c3c3b9f`
- License SHA-256: `ec9fb67dcb25eccc416ed56e1aab819222c805a2a4bfe4cb19e7556bf2ffde80`

To update, download an explicitly pinned package with `npm pack --ignore-scripts`,
verify its integrity, copy these two files, update the loader path and checksums,
and run `web/e2e/cypress/e2e/console.cy.js` against the actual bundle. Supported
browser requirements are documented in [Mermaid's usage guide](https://mermaid.js.org/config/usage.html).

The console lazy-loads this asset for fenced Mermaid blocks and uses
[strict mode](https://mermaid.js.org/config/schema-docs/config.html#securitylevel).
Model-supplied directives/frontmatter and `@{...}` shape metadata are shown as
source instead: the latter permits image fetching before sanitization. Classic
text-only sequence/flowchart syntax remains supported. Failed rendering leaves
the answer and its original diagram source readable.
