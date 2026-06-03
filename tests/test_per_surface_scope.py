"""Phase 4 evolutionary-TDD walk for per-surface closure application.

Each query surface honors the closure that the chokepoint applies.
Where Phase 2 verified the wrapper's contract in isolation and Phase 3
verified the resolve-and-wrap step at the request boundary, Phase 4
verifies the end-to-end semantics through each public surface (search,
themes, graph, context-boost, per-file tools).

This file grows one demand at a time. Each cycle adds a new behavioral
demand to ``TestPerSurfaceScope``; the test file *is* the spec for the
end-to-end closure semantics.

Fixture nomenclature: ``shared`` is a leaf library; ``product`` depends
on ``shared``; ``extension`` depends on ``product``.
"""
from __future__ import annotations

import asyncio
from pathlib import Path


def _write_config(tmp_path: Path, body: str) -> Path:
    p = tmp_path / 'ariadne.yaml'
    p.write_text(body, encoding='utf-8')
    return p


def _make_service(tmp_path: Path, cfg_body: str):
    from config import Config
    from library import Library
    from ariadne_mcp.service import AriadneService

    cfg_path = _write_config(tmp_path, cfg_body)
    cfg = Config(cfg_path)
    library = Library(tmp_path / 'library.db')
    svc = AriadneService()
    svc._config = cfg
    svc._library = library
    return svc, library


class TestPerSurfaceScope:
    # ---- T1 -----------------------------------------------------------
    # ``search(source='product')`` returns only docs in the closure.
    # The original culprit (mcp_service_search._search_uncached calling
    # list_documents_lite() with no scope) leaked sibling-source docs
    # into the response. With search refactored to use _resolve_scope,
    # an extension doc that would otherwise match cannot surface.
    #
    # Exercises the empty-query path so the test doesn't need a live
    # embedding service — the closure filter applies before any ranking,
    # so the property holds regardless of ranking mode.
    def test_t1_search_scopes_by_source(self, tmp_path: Path) -> None:
        svc, library = _make_service(tmp_path, '''\
sources:
  shared:
    path: /tmp/shared
  product:
    path: /tmp/product
    depends_on: [shared]
  extension:
    path: /tmp/extension
    depends_on: [product]
''')
        library.add_document(
            content_type='explanation', title='product-auth-guide',
            content='auth in product', source_name='product',
        )
        library.add_document(
            content_type='explanation', title='shared-auth-base',
            content='auth in shared', source_name='shared',
        )
        library.add_document(
            content_type='explanation', title='extension-auth-flow',
            content='auth in extension', source_name='extension',
        )

        result = asyncio.run(svc.search(query='', source='product'))

        titles = {d.title for d in result.documents}
        assert titles == {'product-auth-guide', 'shared-auth-base'}
        assert 'extension-auth-flow' not in titles

    # ---- T2 -----------------------------------------------------------
    # The same search path respects the directional flip end-to-end:
    # from the leaf source's perspective (``source='shared'``) the
    # reverse closure ``{shared, product, extension}`` is in effect, so
    # all three docs surface. No per-source policy code in the search
    # path — the closure rule from Config.scope_closure does it.
    def test_t2_search_from_leaf_returns_reverse_closure(
        self, tmp_path: Path,
    ) -> None:
        svc, library = _make_service(tmp_path, '''\
sources:
  shared:
    path: /tmp/shared
  product:
    path: /tmp/product
    depends_on: [shared]
  extension:
    path: /tmp/extension
    depends_on: [product]
''')
        library.add_document(
            content_type='explanation', title='product-auth-guide',
            content='auth in product', source_name='product',
        )
        library.add_document(
            content_type='explanation', title='shared-auth-base',
            content='auth in shared', source_name='shared',
        )
        library.add_document(
            content_type='explanation', title='extension-auth-flow',
            content='auth in extension', source_name='extension',
        )

        leaf = asyncio.run(svc.search(query='', source='shared'))
        leaf_titles = {d.title for d in leaf.documents}
        assert leaf_titles == {
            'product-auth-guide', 'shared-auth-base',
            'extension-auth-flow',
        }

        # T1 still holds from this fixture.
        mid = asyncio.run(svc.search(query='', source='product'))
        mid_titles = {d.title for d in mid.documents}
        assert mid_titles == {'product-auth-guide', 'shared-auth-base'}
        assert 'extension-auth-flow' not in mid_titles

    # ---- T3 (rescinded) -----------------------------------------------
    # Earlier this cycle asserted that ``themes_action`` filtered theme
    # members by closure. That demand was retracted: themes are
    # library-internal cross-source data (theme docs are created without
    # ``source_name`` by ``docgen/cluster.py``), so routing themes_action
    # through ScopedLibrary would filter out every theme that has no
    # source_name. The architectural decision
    # "Library-internal modules — legitimately unscoped" is that themes
    # operate below the chokepoint, using the raw library.
    #
    # The test body below is retained as a marker for the rescinded
    # demand. It pins the *new* behavior: themes_action takes raw
    # library, returns themes regardless of caller scope.
    def test_t3_themes_action_is_library_internal(
        self, tmp_path: Path,
    ) -> None:
        from library import Library
        from ariadne_mcp.service_themes import themes_action

        with Library(tmp_path / 'library.db') as library:
            # Theme summary doc with source_name=None (the production
            # case — docgen/cluster.py creates theme docs without
            # source_name because themes are cross-source by design).
            theme_doc = library.add_document(
                content_type='theme', title='cross-source-theme-summary',
                content='theme spans multiple sources',
            )
            product_member = library.add_document(
                content_type='explanation', title='product-member',
                content='product side', source_name='product',
            )
            ext_member = library.add_document(
                content_type='explanation', title='extension-member',
                content='extension side', source_name='extension',
            )
            library.add_theme(
                cluster_id='cluster-1', doc_id=theme_doc.id,
                member_count=2, resolution=1.0,
                summary_hash='hash', coherent=True, dirty=False,
            )
            library.set_theme_members(
                'cluster-1',
                [(product_member.id, 1.0), (ext_member.id, 1.0)],
            )

            # themes_action takes the raw library — themes are
            # library-internal. ``list`` returns the theme regardless
            # of caller scope; ``members`` returns every member.
            list_resp = themes_action(
                library, action='list',
            )
            cluster_ids = {
                t['cluster_id'] for t in list_resp['themes']
            }
            assert 'cluster-1' in cluster_ids

            members_resp = themes_action(
                library, action='members', cluster_id='cluster-1',
            )
            member_titles = {
                m['title'] for m in members_resp['members']
            }
            assert member_titles == {'product-member', 'extension-member'}

    # ---- T4 -----------------------------------------------------------
    # SCIP graph surface end-to-end. Construct the scope via the same
    # resolver the request boundary uses, then verify
    # ``scip_callers``/``scip_callees`` honor the directional rule:
    #
    #   - from product's scope, callers of ``shared.authenticate``
    #     surface only product's call site (extension's is dropped —
    #     both endpoints' source_name must be in closure).
    #   - from shared's scope (leaf → reverse closure
    #     ``{shared, product, extension}``), both product's and
    #     extension's call sites surface.
    def test_t4_scip_graph_surface_respects_directional_closure(
        self, tmp_path: Path,
    ) -> None:
        from config import Config
        from library import Library
        from scope_resolution import make_scoped_library

        cfg_path = _write_config(tmp_path, '''\
sources:
  shared:
    path: /tmp/shared
  product:
    path: /tmp/product
    depends_on: [shared]
  extension:
    path: /tmp/extension
    depends_on: [product]
''')
        cfg = Config(cfg_path)
        with Library(tmp_path / 'library.db') as library:
            with library._conn_provider.acquire() as conn:
                symbols = [
                    ('shared:authenticate', 'shared', 'python',
                     'shared/auth.py', 10, 20,
                     'function', 'authenticate',
                     'shared.authenticate', None),
                    ('product:handle', 'product', 'python',
                     'product/main.py', 30, 40,
                     'function', 'handle',
                     'product.handle', None),
                    ('extension:bootstrap', 'extension', 'python',
                     'extension/init.py', 5, 15,
                     'function', 'bootstrap',
                     'extension.bootstrap', None),
                ]
                conn.executemany(
                    'INSERT INTO scip_symbols VALUES '
                    '(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    symbols,
                )
                edges = [
                    ('product:handle', 'shared:authenticate', 'call',
                     'product/main.py', 35, 'precise'),
                    ('extension:bootstrap', 'shared:authenticate',
                     'call', 'extension/init.py', 12, 'precise'),
                ]
                conn.executemany(
                    'INSERT INTO scip_edges VALUES '
                    '(?, ?, ?, ?, ?, ?)',
                    edges,
                )
                conn.commit()

            # Product scope — only product call site of shared.authenticate.
            product_scope = make_scoped_library(cfg, library, 'product')
            from_product = product_scope.scip_callers(
                'shared:authenticate',
            )
            assert {e.caller.source_name for e in from_product} == {
                'product',
            }
            assert all(
                e.callee.source_name == 'shared' for e in from_product
            )

            # Leaf source — reverse closure; both callers surface.
            shared_scope = make_scoped_library(cfg, library, 'shared')
            from_shared = shared_scope.scip_callers(
                'shared:authenticate',
            )
            assert {e.caller.source_name for e in from_shared} == {
                'product', 'extension',
            }

            # Extension scope — forward-closes to product+shared+
            # extension; both callers surface (this is the top-of-chain
            # scope which by construction includes everything below).
            extension_scope = make_scoped_library(
                cfg, library, 'extension',
            )
            from_extension = extension_scope.scip_callers(
                'shared:authenticate',
            )
            assert {e.caller.source_name for e in from_extension} == {
                'product', 'extension',
            }

    # ---- T5 -----------------------------------------------------------
    # Context-boost respects the closure. The search path boosts docs
    # whose ``source_files`` reference the provided ``context_file``,
    # plus their graph neighbors. Both lookups go through the
    # ScopedLibrary, so an extension-side doc claiming to document the
    # same file as a product doc can't be promoted into a product-scope
    # search result.
    def test_t5_context_boost_does_not_leak_out_of_closure(
        self, tmp_path: Path,
    ) -> None:
        svc, library = _make_service(tmp_path, '''\
sources:
  shared:
    path: /tmp/shared
  product:
    path: /tmp/product
    depends_on: [shared]
  extension:
    path: /tmp/extension
    depends_on: [product]
''')
        # The product doc references 'context.py'; so does an extension
        # doc. Without closure filtering, the extension doc would be in
        # context_boost_ids — but since the boost set is now produced
        # by the scoped library, the extension doc can't surface.
        library.add_document(
            content_type='explanation', title='product-doc-on-context',
            content='product touches context.py', source_name='product',
            source_files=['product/context.py'],
        )
        library.add_document(
            content_type='explanation',
            title='extension-doc-on-context',
            content='extension touches context.py',
            source_name='extension',
            source_files=['extension/context.py'],
        )
        # A no-op product doc so the search has results overall.
        library.add_document(
            content_type='explanation', title='product-other',
            content='other product content', source_name='product',
        )

        result = asyncio.run(svc.search(
            query='',
            source='product',
            context_file='context.py',
        ))

        titles = {d.title for d in result.documents}
        assert 'product-doc-on-context' in titles
        assert 'extension-doc-on-context' not in titles

    # ---- T6 -----------------------------------------------------------
    # Per-file tool composition. A typical per-file tool (think
    # ``ariadne_explain``) walks: find docs whose source_files reference
    # the path → fetch their related docs → batch-load full content.
    # When the tool runs against a closure-scoped library view, every
    # step must drop out-of-closure rows: the seed find, the
    # ``get_related`` neighbors, and the final batch.
    #
    # This cycle simulates that walk through ScopedLibrary primitives.
    # The plan's per-file tools (``explain``, ``impact_radius``, etc.)
    # still call ``Library`` methods directly; their migration to call
    # the scoped wrapper is the same mechanical sweep as Phase 3's
    # deferred T7 — covered structurally by Phase 5's lint check.
    def test_t6_per_file_tool_composition_stays_scoped(
        self, tmp_path: Path,
    ) -> None:
        from config import Config
        from library import Library
        from scope_resolution import make_scoped_library

        cfg_path = _write_config(tmp_path, '''\
sources:
  shared:
    path: /tmp/shared
  product:
    path: /tmp/product
    depends_on: [shared]
  extension:
    path: /tmp/extension
    depends_on: [product]
''')
        cfg = Config(cfg_path)
        with Library(tmp_path / 'library.db') as library:
            # Seed: a product doc and an extension doc both reference
            # 'feature.py'. Without the closure filter, both would
            # surface as the "per-file" seed set.
            product_doc = library.add_document(
                content_type='explanation', title='product-feature',
                content='product explanation of feature.py',
                source_name='product',
                source_files=['product/feature.py'],
            )
            ext_doc = library.add_document(
                content_type='explanation',
                title='extension-feature',
                content='extension explanation of feature.py',
                source_name='extension',
                source_files=['extension/feature.py'],
            )
            # Cross-refs: the product doc links to a product-side
            # detail doc AND to an extension-side detail doc. Only
            # the product-side neighbor should surface.
            product_detail = library.add_document(
                content_type='explanation', title='product-detail',
                content='product detail', source_name='product',
            )
            ext_detail = library.add_document(
                content_type='explanation', title='extension-detail',
                content='extension detail', source_name='extension',
            )
            with library._conn_provider.acquire() as conn:
                conn.executemany(
                    'INSERT INTO doc_graph(source_id, target_id, '
                    'edge_type, weight) VALUES (?, ?, ?, ?)',
                    [
                        (product_doc.id, product_detail.id,
                         'related', 1.0),
                        (product_doc.id, ext_detail.id,
                         'related', 1.0),
                        (ext_doc.id, ext_detail.id,
                         'related', 1.0),
                    ],
                )
                conn.commit()

            scoped = make_scoped_library(cfg, library, 'product')

            # Step 1: seed find — only product-side doc surfaces.
            seeds = scoped.find_documents_by_source_files(['feature.py'])
            seed_titles = {d.title for d in seeds}
            assert seed_titles == {'product-feature'}

            # Step 2: cross-refs from the seed — only product-side
            # neighbor surfaces.
            related = scoped.get_related(product_doc.id)
            assert {r['id'] for r in related} == {product_detail.id}

            # Step 3: final batch — out-of-closure ids dropped from
            # the result silently, never present.
            ids_under_test = [
                product_doc.id, ext_doc.id,
                product_detail.id, ext_detail.id,
            ]
            batch = scoped.get_documents_batch(ids_under_test)
            assert {d.title for d in batch} == {
                'product-feature', 'product-detail',
            }
            assert 'extension-feature' not in {d.title for d in batch}
            assert 'extension-detail' not in {d.title for d in batch}
