'use strict';

/*
 * Regression test for cross-version @vue/compiler-sfc handling.
 *
 * The extractor must support BOTH calling conventions:
 *   Vue 3:   parse(source, { filename })  -> { descriptor, errors }   (loc.start.line provided)
 *   Vue 2.7: parse({ source, filename })  -> SFCDescriptor directly    (char offset only, no loc)
 *
 * We stub `sfc` for each shape so the test is hermetic — no real Vue
 * install required. Run: `node scripts/scip/extract-vue-scripts.test.js`
 */
const assert = require('node:assert');
const {
  parseSfc,
  startLineOf,
  buildCompanion,
} = require('./extract-vue-scripts.js');

const SRC =
  '<template>\n' + //                 line 1
  '  <div/>\n' + //                   line 2
  '</template>\n' + //                line 3
  '<script lang="ts">\n' + //         line 4
  'export default {}\n' + //          line 5
  '</script>\n'; //                   line 6

// The <script> block content starts at the newline right after the
// opening tag (end of line 4); its first code line is line 5.
const SCRIPT_CONTENT = '\nexport default {}\n';
const SCRIPT_OFFSET = SRC.indexOf('export default {}') - 1; // the '\n' before it

// Vue 3: parse(source, {filename}) -> { descriptor, errors }, loc provided.
const vue3 = {
  parse(source, opts) {
    assert.strictEqual(typeof source, 'string', 'vue3 parse takes source string');
    assert.ok(opts && opts.filename, 'vue3 parse takes {filename}');
    return {
      errors: [],
      descriptor: {
        script: {
          content: SCRIPT_CONTENT,
          attrs: { lang: 'ts' },
          loc: { start: { line: 4 } },
        },
        scriptSetup: null,
      },
    };
  },
};

// Vue 2.7: positional-source form is ignored (no .descriptor); the
// options-object form returns the SFCDescriptor directly with a char
// offset (`start`) and no `loc`.
const vue2 = {
  parse(arg) {
    if (typeof arg === 'string') {
      return { errors: [] }; // no .descriptor, no .script — the bug trigger
    }
    assert.ok(arg && typeof arg.source === 'string' && arg.filename,
      'vue2 parse takes {source, filename}');
    return {
      errors: [],
      script: {
        content: SCRIPT_CONTENT,
        attrs: { lang: 'ts' },
        start: SCRIPT_OFFSET,
      },
      scriptSetup: null,
    };
  },
};

// parseSfc yields a descriptor with .script for BOTH versions.
for (const [name, sfc] of [['vue3', vue3], ['vue2', vue2]]) {
  const { descriptor } = parseSfc(sfc, SRC, 'X.vue');
  assert.ok(descriptor && descriptor.script, `${name}: descriptor.script present`);
  assert.strictEqual(descriptor.script.attrs.lang, 'ts', `${name}: lang preserved`);
}

// startLineOf agrees across versions: both resolve to original line 4.
for (const [name, sfc] of [['vue3', vue3], ['vue2', vue2]]) {
  const block = parseSfc(sfc, SRC, 'X.vue').descriptor.script;
  assert.strictEqual(startLineOf(block, SRC), 4, `${name}: startLine == 4`);
}

// buildCompanion preserves line fidelity: `export default {}` lands on
// original line 5 (0-indexed 4) for both versions.
for (const [name, sfc] of [['vue3', vue3], ['vue2', vue2]]) {
  const block = parseSfc(sfc, SRC, 'X.vue').descriptor.script;
  const companion = buildCompanion([
    { content: block.content, startLine: startLineOf(block, SRC) },
  ]);
  const lines = companion.split('\n');
  assert.ok(lines[4].includes('export default {}'),
    `${name}: export on original line 5, got ${JSON.stringify(lines[4])}`);
}

console.log('extract-vue-scripts: all assertions passed (vue2 + vue3)');
