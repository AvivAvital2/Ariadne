'use strict';
/*
 * Functional tests for web/static/console.html (the Ask console).
 *
 * These load the REAL page in a real DOM (jsdom), inject a mocked `fetch`, and
 * drive the actual click/keyboard handlers — asserting on the rendered DOM and
 * the HTTP calls the page makes. No sockets: jsdom needs none, and fetch is
 * stubbed. This is functional evidence the frontend behaves as intended.
 *
 * Run:  NODE_PATH=<a node_modules with jsdom> node --test tests/test_console_ui.js
 * (Skips cleanly if jsdom isn't resolvable.)
 */
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

let JSDOM;
try { ({ JSDOM } = require('jsdom')); } catch { /* skip below */ }
const skip = JSDOM ? false : 'jsdom not resolvable (set NODE_PATH or npm i -D jsdom)';

const HTML = fs.readFileSync(
  path.join(__dirname, '..', 'web', 'static', 'console.html'), 'utf8');

// Canned MCP-tool responses — shapes match ariadne_mcp/models.py
// (AskResponse{answer,confidence,sources,event_id}; SearchResponse
//  documents[DocumentResult{title,content_type,source_files,score}], …).
const ASK = {
  answer: 'Retries live in **http/resilience.py** via the `@retryable` decorator.',
  confidence: 'high', sources: ['The @retryable decorator'], event_id: 42,
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

function makeDom(opts = {}) {
  const calls = [];
  const routes = {
    '/api/sources': { default_source: 'proj', sources: [{ name: 'proj' }] },
    '/api/ask': ASK, '/api/search': SEARCH,
    '/api/feedback': { success: true, message: 'Hit logged.' },
  };
  const fetchImpl = (url, o = {}) => {
    let body = {}; try { body = o.body ? JSON.parse(o.body) : {}; } catch {}
    calls.push({ url, method: o.method || 'GET', body });
    if (opts.reject && opts.reject[url]) {
      return Promise.resolve({ ok: false, status: 500, json: () => Promise.resolve({ error: opts.reject[url] }) });
    }
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(routes[url] ?? {}) });
  };
  const dom = new JSDOM(HTML, {
    runScripts: 'dangerously',
    beforeParse(window) { window.fetch = fetchImpl; },
  });
  return { window: dom.window, doc: dom.window.document, calls };
}

const tick = () => new Promise(r => setTimeout(r, 0));
async function waitFor(fn, ms = 1500) {
  const end = Date.now() + ms;
  while (Date.now() < end) { if (fn()) return true; await tick(); }
  return fn();
}

test('initial load: source pill + default chips, /api/sources fetched', { skip }, async () => {
  const { doc, calls } = makeDom();
  await waitFor(() => doc.getElementById('srcName').textContent === 'proj');
  assert.equal(doc.getElementById('srcName').textContent, 'proj');
  assert.ok(doc.querySelectorAll('#chips .chip').length >= 1, 'default chips render');
  assert.ok(calls.some(c => c.url === '/api/sources'), 'fetched /api/sources on load');
});

test('ask: posts to /api/ask + /api/search, renders answer + ranked rail + chips', { skip }, async () => {
  const { doc, calls } = makeDom();
  doc.getElementById('q').value = 'how are retries handled?';
  doc.getElementById('askBtn').click();
  await waitFor(() => !doc.getElementById('answerGrid').hidden);

  const ask = calls.find(c => c.url === '/api/ask');
  const search = calls.find(c => c.url === '/api/search');
  assert.equal(ask.method, 'POST');
  assert.deepEqual(ask.body, { question: 'how are retries handled?', role: 'developer' });
  assert.equal(search.method, 'POST');
  assert.deepEqual(search.body, { query: 'how are retries handled?', role: 'developer' });

  const card = doc.getElementById('answerCard');
  assert.ok(card.innerHTML.includes('http/resilience.py'), 'answer prose rendered (**bold** → text)');
  assert.match(card.innerHTML, /high confidence/i, 'confidence badge from AskResponse.confidence');
  assert.ok(/explanation|gotcha/.test(card.innerHTML), 'doc-type badges derived from search docs');
  assert.match(card.textContent, /retryable/, 'inline `code` rendered');
  assert.match(card.innerHTML, /spool/, 'improvement_hint nudge shown');

  const rail = doc.getElementById('rail').innerHTML;
  assert.ok(rail.includes('The @retryable decorator'), 'source title in rail');
  assert.ok(rail.includes('0.94'), 'match score surfaced in rail');
  assert.ok(rail.includes('http/resilience.py'), 'source file path in rail');

  const chips = [...doc.querySelectorAll('#chips .chip')].map(c => c.textContent);
  assert.ok(chips.includes('What breaks if I change RetryPolicy?'), 'chips from suggested_queries');
});

test('feedback: 👍 posts /api/feedback with the ask event_id', { skip }, async () => {
  const { doc, calls } = makeDom();
  doc.getElementById('q').value = 'q';
  doc.getElementById('askBtn').click();
  await waitFor(() => !doc.getElementById('answerGrid').hidden);
  const up = [...doc.querySelectorAll('#answerCard .fb button')].find(b => b.dataset.fb === '1');
  up.click();
  await waitFor(() => up.disabled);
  const fb = calls.find(c => c.url === '/api/feedback');
  assert.deepEqual(fb.body, { event_id: 42, helpful: true });
  assert.match(up.textContent, /Logged hit/);
});

test('role toggle: Product re-asks with role=product_manager', { skip }, async () => {
  const { doc, calls } = makeDom();
  doc.getElementById('q').value = 'q1';
  doc.getElementById('askBtn').click();
  await waitFor(() => !doc.getElementById('answerGrid').hidden);
  [...doc.querySelectorAll('#roleSeg button')].find(b => b.dataset.role === 'product_manager').click();
  // wait for the re-ask to actually re-render (not just for the fetch to fire)
  await waitFor(() => /class="badge role">product</.test(doc.getElementById('answerCard').innerHTML));
  const asks = calls.filter(c => c.url === '/api/ask');
  assert.ok(asks.length >= 2, 'toggling role re-asked');
  assert.equal(asks[asks.length - 1].body.role, 'product_manager', 'role reaches /api/ask');
  assert.match(doc.getElementById('answerCard').innerHTML, /class="badge role">product</, 'product role badge rendered');
});

test('error: a failing /api/ask shows an error and hides the answer', { skip }, async () => {
  const { doc } = makeDom({ reject: { '/api/ask': 'model unavailable' } });
  doc.getElementById('q').value = 'q';
  doc.getElementById('askBtn').click();
  await waitFor(() => {
    const s = doc.getElementById('status');
    return !s.hidden && s.className.includes('err');
  });
  assert.match(doc.getElementById('status').textContent, /Could not answer/);
  assert.ok(doc.getElementById('answerGrid').hidden, 'answer grid stays hidden on error');
});

test('onboarding: "ready" screen links into the Ask console (Phase 3 handoff)', { skip }, async () => {
  const onb = fs.readFileSync(path.join(__dirname, '..', 'web', 'static', 'onboarding.html'), 'utf8');
  const dom = new JSDOM(onb, {
    runScripts: 'dangerously',
    beforeParse(w) { w.fetch = () => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({}) }); },
  });
  const doc = dom.window.document;
  const cta = doc.querySelector('#gen-ready a[href="/static/console.html"]');
  assert.ok(cta, 'the ready screen has an Ask-console handoff link');
  assert.match(cta.textContent, /Ask/);
  assert.equal(typeof dom.window.startBuild, 'function', 'Step-5 build handler (startBuild) is wired');
});
