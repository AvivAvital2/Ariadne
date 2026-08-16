"""Integration test A — full topology, end-to-end isolation.

A single test that exercises every public surface against the target
topology (shared / product / extension) and asserts the directional
closure rule holds through:

  - search (`mcp_service_search`)
  - themes (`themes_action` via `mcp_service_themes`)
  - SCIP graph (`scip_callers` via `ScopedLibrary`)
  - list_all (`mcp_service_admin`)
  - error paths (fail-closed when no source resolves)

Topology:

    shared (leaf)  ←─ product (depends_on shared)  ←─ extension (depends_on product)

Closures (by the directional rule):

    shared    → {shared, product, extension}   (leaf → reverse)
    product   → {product, shared}              (forward)
    extension → {extension, product, shared}   (forward)

This file does NOT grow one demand at a time — it's the cross-phase
acceptance gate. Each test method exercises one surface end-to-end.
The fixtures are shared across methods.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Shared fixture: build a Library populated with the topology, plus a
# Config that knows the dep graph, plus a service wired to both.
# ---------------------------------------------------------------------------


@pytest.fixture
def topology(tmp_path: Path):
    """Build the (shared, product, extension) topology."""
    from config import Config
    from library import Library
    from ariadne_mcp.service import AriadneService

    # Real source directories so cwd auto-detection works for the
    # fail-closed test.
    shared_dir = tmp_path / 'src' / 'shared'
    product_dir = tmp_path / 'src' / 'product'
    extension_dir = tmp_path / 'src' / 'extension'
    for d in (shared_dir, product_dir, extension_dir):
        d.mkdir(parents=True)

    cfg_path = tmp_path / 'ariadne.yaml'
    cfg_path.write_text(f'''\
sources:
  shared:
    path: {shared_dir}
  product:
    path: {product_dir}
    depends_on: [shared]
  extension:
    path: {extension_dir}
    depends_on: [product]
''', encoding='utf-8')
    cfg = Config(cfg_path)

    library = Library(tmp_path / 'library.db')

    # Seed docs across all three sources covering a small auth domain.
    shared_auth = library.add_document(
        content_type='explanation',
        title='shared-auth-base',
        content='auth primitives in the shared library',
        source_name='shared',
        source_files=['shared/auth.py'],
    )
    product_auth = library.add_document(
        content_type='explanation',
        title='product-auth-middleware',
        content='product wraps shared.authenticate for HTTP requests',
        source_name='product',
        source_files=['product/middleware.py'],
    )
    extension_auth = library.add_document(
        content_type='explanation',
        title='extension-auth-flow',
        content='extension chains product.handle for SSO',
        source_name='extension',
        source_files=['extension/sso.py'],
    )

    # SCIP symbols + edges for the cross-source graph.
    with library._conn_provider.acquire() as conn:
        conn.executemany(
            'INSERT INTO scip_symbols (canonical_id, source_name, language, file, line_start, line_end, kind, display_name, qualified_name, parent_qualified_name) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            [
                ('shared:authenticate', 'shared', 'python',
                 'shared/auth.py', 10, 20, 'function',
                 'authenticate', 'shared.authenticate', None),
                ('product:middleware', 'product', 'python',
                 'product/middleware.py', 30, 40, 'function',
                 'middleware', 'product.middleware', None),
                ('extension:sso', 'extension', 'python',
                 'extension/sso.py', 5, 15, 'function',
                 'sso', 'extension.sso', None),
            ],
        )
        conn.executemany(
            'INSERT INTO scip_edges VALUES '
            '(?, ?, ?, ?, ?, ?)',
            [
                ('product:middleware', 'shared:authenticate', 'call',
                 'product/middleware.py', 35, 'precise'),
                ('extension:sso', 'shared:authenticate', 'call',
                 'extension/sso.py', 12, 'precise'),
                ('extension:sso', 'product:middleware', 'call',
                 'extension/sso.py', 13, 'precise'),
            ],
        )
        conn.commit()

    # Theme with cross-source members (product summary, members from
    # product + extension). Used by the themes-surface test.
    theme_doc = library.add_document(
        content_type='theme',
        title='product-auth-theme-summary',
        content='cross-source auth flow',
        source_name='product',
    )
    library.add_theme(
        cluster_id='cluster-auth',
        doc_id=theme_doc.id,
        member_count=2,
        resolution=1.0,
        summary_hash='hash-auth',
        coherent=True,
        dirty=False,
    )
    library.set_theme_members(
        'cluster-auth',
        [(product_auth.id, 1.0), (extension_auth.id, 1.0)],
    )

    # Build the service with our config + library wired in (bypass
    # the singleton).
    svc = AriadneService()
    svc._config = cfg
    svc._library = library

    yield {
        'svc': svc,
        'library': library,
        'cfg': cfg,
        'tmp_path': tmp_path,
        'shared_dir': shared_dir,
        'product_dir': product_dir,
        'extension_dir': extension_dir,
        'shared_auth_id': shared_auth.id,
        'product_auth_id': product_auth.id,
        'extension_auth_id': extension_auth.id,
    }

    library.close()


class TestTopologyIsolation:
    # ---- demand 1: search from product's scope ------------------------
    def test_search_from_product_excludes_extension(self, topology) -> None:
        result = asyncio.run(
            topology['svc'].search(query='', source='product'),
        )
        titles = {d.title for d in result.documents}
        assert 'product-auth-middleware' in titles
        assert 'shared-auth-base' in titles
        assert 'extension-auth-flow' not in titles

    # ---- demand 2: search from shared's scope (reverse closure) -------
    def test_search_from_leaf_returns_reverse_closure(
        self, topology,
    ) -> None:
        result = asyncio.run(
            topology['svc'].search(query='', source='shared'),
        )
        titles = {d.title for d in result.documents}
        # All three auth docs surface (leaf → reverse closure picks up
        # every consumer). The theme summary doc (source='product') is
        # also in the closure and may or may not surface depending on
        # search ranking; we only assert the directional flip works.
        assert 'shared-auth-base' in titles
        assert 'product-auth-middleware' in titles
        assert 'extension-auth-flow' in titles

    # ---- demand 3: search from extension's scope ----------------------
    def test_search_from_extension_returns_full_stack(
        self, topology,
    ) -> None:
        result = asyncio.run(
            topology['svc'].search(query='', source='extension'),
        )
        titles = {d.title for d in result.documents}
        # extension forward-closes to product + shared, plus itself.
        # All three sources should surface.
        assert 'shared-auth-base' in titles
        assert 'product-auth-middleware' in titles
        assert 'extension-auth-flow' in titles

    # ---- demand 4: SCIP callers from shared's scope (reverse) ---------
    def test_scip_callers_from_leaf_sees_full_consumer_panorama(
        self, topology,
    ) -> None:
        from scope_resolution import make_scoped_library

        scoped = make_scoped_library(
            topology['cfg'], topology['library'], 'shared',
        )
        edges = scoped.scip_callers('shared:authenticate')
        callers = {e.caller.source_name for e in edges}
        assert callers == {'product', 'extension'}

    # ---- demand 5: SCIP callers from product's scope (forward) --------
    def test_scip_callers_from_product_only_product_call_sites(
        self, topology,
    ) -> None:
        from scope_resolution import make_scoped_library

        scoped = make_scoped_library(
            topology['cfg'], topology['library'], 'product',
        )
        edges = scoped.scip_callers('shared:authenticate')
        callers = {e.caller.source_name for e in edges}
        # Extension's call site is in extension's source, which is NOT
        # in product's forward closure → dropped.
        assert callers == {'product'}

    # ---- demand 6 (rescinded): themes are library-internal ------------
    # Earlier this demand asserted themes_action filtered members by
    # closure. That was retracted after the review caught that theme
    # docs created by ``docgen/cluster.py`` have ``source_name=None``,
    # so any scoped read would filter them all out. Themes operate
    # below the chokepoint (see designs/directional-closure-scoping.md
    # § "Library-internal modules — legitimately unscoped"). The test
    # now pins the new behavior: themes_action takes raw library and
    # returns themes regardless of any caller scope.
    def test_themes_action_is_library_internal(self, topology) -> None:
        from ariadne_mcp.service_themes import themes_action

        list_resp = themes_action(topology['library'], action='list')
        cluster_ids = {t['cluster_id'] for t in list_resp['themes']}
        assert 'cluster-auth' in cluster_ids

        members_resp = themes_action(
            topology['library'], action='members',
            cluster_id='cluster-auth',
        )
        titles = {m['title'] for m in members_resp['members']}
        # Both members surface — themes are intentionally cross-source.
        assert titles == {
            'product-auth-middleware', 'extension-auth-flow',
        }

    # ---- demand 7a: misspelled source raises a structured error -------
    def test_misspelled_source_raises_structured_error(
        self, topology,
    ) -> None:
        from scope_resolution import make_scoped_library

        # make_scoped_library unifies the fail-closed signal as
        # LookupError (KeyError is a LookupError subclass but the
        # contract to Claude in mcp_server.py:96+ says LookupError).
        with pytest.raises(LookupError) as exc:
            make_scoped_library(
                topology['cfg'], topology['library'], 'prodcut',  # typo
            )
        message = str(exc.value)
        # Error names the offender AND lists configured sources.
        assert 'prodcut' in message
        assert 'product' in message
        assert 'shared' in message

    # ---- demand 7b: fail-closed when no source resolves ---------------
    def test_no_source_fail_closed(self, topology, monkeypatch) -> None:
        # cwd outside any configured source; no default_source in cfg.
        unrelated = topology['tmp_path'] / 'unrelated'
        unrelated.mkdir()
        monkeypatch.chdir(unrelated)

        with pytest.raises(LookupError) as exc:
            asyncio.run(topology['svc'].list_all())
        message = str(exc.value)
        assert 'no source' in message.lower()
        assert (
            'configured' in message.lower()
            or 'project tree' in message.lower()
        )
