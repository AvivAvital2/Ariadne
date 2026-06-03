"""Phase 3 evolutionary-TDD walk for request-boundary scoping.

Each entry point (MCP tool, CLI handler) must resolve a source, compute
the closure, build a ``ScopedLibrary``, and use it. There is no path
into business logic that holds a raw ``Library``.

This file grows one demand at a time. Each cycle adds a new behavioral
demand to ``TestRequestBoundary``; the test file *is* the spec for the
resolve-and-wrap discipline at the request boundary.

Fixture nomenclature: ``shared`` is a leaf shared library; ``product``
depends on ``shared``; ``extension`` depends on ``product``.
"""
from __future__ import annotations

from pathlib import Path


def _write_config(tmp_path: Path, body: str) -> Path:
    p = tmp_path / 'ariadne.yaml'
    p.write_text(body, encoding='utf-8')
    return p


def _make_service(tmp_path: Path, cfg_body: str):
    """Build an isolated AriadneService for the test.

    Bypasses the module-level singleton: we instantiate the class
    directly, point ``_config`` at a fresh Config, and ``_library`` at
    a fresh Library that lives in ``tmp_path``.
    """
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


class TestRequestBoundary:
    # ---- T1 -----------------------------------------------------------
    # A handler invoked with an explicit ``source='product'`` resolves
    # that to closure ``{'product', 'shared'}``, builds a ScopedLibrary,
    # and uses it. The end-to-end observation: a response from the
    # handler omits out-of-closure rows that *would* surface if the
    # handler had used the raw Library.
    def test_t1_explicit_source_scopes_the_response(
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
        # Two product docs and one extension doc. With source='product',
        # the closure is {product, shared}, so the extension doc must
        # be absent from the response.
        library.add_document(
            content_type='explanation', title='product-doc-a',
            content='product content a', source_name='product',
        )
        library.add_document(
            content_type='explanation', title='product-doc-b',
            content='product content b', source_name='product',
        )
        library.add_document(
            content_type='explanation', title='extension-doc',
            content='extension content', source_name='extension',
        )

        response = svc.list_all(source='product')

        titles = {d.title for d in response.documents}
        assert titles == {'product-doc-a', 'product-doc-b'}
        assert 'extension-doc' not in titles

    # ---- T2 -----------------------------------------------------------
    # MCP fail-closed: the MCP service does NOT auto-detect the source
    # from cwd. The real server's process cwd is the Ariadne install (it's
    # launched with a pinned ``--directory``), not the user's project, so
    # cwd detection would always mis-resolve to whatever source contains
    # Ariadne — the production bug where a projecta question was answered from
    # the ``ariadne`` source. With no explicit ``source`` the request
    # fails closed (LookupError); the source comes from the caller's
    # decomposition via the explicit ``source`` argument (as in T1).
    def test_t2_mcp_fails_closed_without_explicit_source(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        import pytest

        # Real source directories so Config.get_source_scope can match
        # cwd against them.
        shared_dir = tmp_path / 'src' / 'shared'
        product_dir = tmp_path / 'src' / 'product'
        extension_dir = tmp_path / 'src' / 'extension'
        for d in (shared_dir, product_dir, extension_dir):
            d.mkdir(parents=True)

        svc, library = _make_service(tmp_path, f'''\
sources:
  shared:
    path: {shared_dir}
  product:
    path: {product_dir}
    depends_on: [shared]
  extension:
    path: {extension_dir}
    depends_on: [product]
''')
        library.add_document(
            content_type='explanation', title='product-doc',
            content='product content', source_name='product',
        )
        library.add_document(
            content_type='explanation', title='extension-doc',
            content='extension content', source_name='extension',
        )

        # Even sitting "inside" the product directory, the MCP service must
        # NOT auto-scope by cwd — trusting the process cwd is what silently
        # answers from the wrong repo. No explicit source => fail closed.
        monkeypatch.chdir(product_dir)
        with pytest.raises(LookupError):
            svc.list_all()  # no source argument — must not guess

        # The source comes from the caller's decomposition (explicit arg),
        # which scopes correctly to {product, shared}.
        response = svc.list_all(source='product')

        titles = {d.title for d in response.documents}
        assert titles == {'product-doc'}
        assert 'extension-doc' not in titles

    # ---- T3 -----------------------------------------------------------
    # Fail-closed: no explicit source, no cwd match, no default. The
    # handler MUST NOT return all documents from all sources. Instead
    # it raises a structured error naming the omission so the caller
    # can pass ``source=`` or run within a configured tree.
    #
    # This is the load-bearing safety property — a missing source
    # argument must never widen the scope to "everything."
    def test_t3_no_resolvable_source_fails_closed(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        import pytest

        # Configure sources whose paths are far from the cwd we'll set,
        # AND don't declare a default_source. With no explicit argument
        # and no cwd match and no default, there is no source to resolve.
        shared_dir = tmp_path / 'src' / 'shared'
        product_dir = tmp_path / 'src' / 'product'
        for d in (shared_dir, product_dir):
            d.mkdir(parents=True)

        svc, library = _make_service(tmp_path, f'''\
sources:
  shared:
    path: {shared_dir}
  product:
    path: {product_dir}
    depends_on: [shared]
''')
        # Add a doc — verifies that nothing leaks even when data exists.
        library.add_document(
            content_type='explanation', title='product-doc',
            content='product content', source_name='product',
        )

        # cwd is in tmp_path itself (not under any source path) — and
        # there is no default_source declared. Both auto-detection paths
        # come up empty.
        unrelated_dir = tmp_path / 'unrelated'
        unrelated_dir.mkdir()
        monkeypatch.chdir(unrelated_dir)

        with pytest.raises(LookupError) as exc:
            svc.list_all()
        message = str(exc.value)
        # Must name the problem AND tell the user how to resolve it.
        # The current confused state (KeyError from scope_closure(None)
        # with a misleading "unknown source None" message) is NOT the
        # right error — a user staring at "unknown source None" can't
        # tell what they did wrong. The fail-closed error must instruct
        # the user to pass ``source=`` or run within a configured tree.
        assert 'no source' in message.lower() or (
            'pass source' in message.lower()
        ), (
            'fail-closed error must guide the user — got: {!r}'.format(
                message,
            )
        )
        assert (
            'configured' in message.lower()
            or 'project tree' in message.lower()
        )

    # ---- T4 -----------------------------------------------------------
    # Multiple handlers share the same resolve-and-wrap discipline. T1
    # proved it for ``list_all``; T4 widens to a second handler whose
    # internal logic touches the documents table in a different way
    # (``coverage`` reads ``source_files`` to compute documented vs
    # undocumented). Same fixture pattern; same closure-filtered outcome.
    #
    # The point of the refinement is structural: if every handler that
    # reads documents goes through ``_resolve_scope``, the fail-closed
    # and closure-filter guarantees from T1-T3 transfer automatically
    # to every refactored entry point.
    def test_t4_coverage_handler_scoped_by_source(
        self, tmp_path: Path,
    ) -> None:
        product_dir = tmp_path / 'src' / 'product'
        product_dir.mkdir(parents=True)
        # Create one real file in product_dir for coverage to enumerate.
        (product_dir / 'tracked.py').write_text('pass\n')
        (product_dir / 'untracked.py').write_text('pass\n')

        # Shared has no path under tmp_path that we need; coverage only
        # looks at the requested source's path. Use a separate dir.
        shared_dir = tmp_path / 'src' / 'shared'
        shared_dir.mkdir(parents=True)
        extension_dir = tmp_path / 'src' / 'extension'
        extension_dir.mkdir(parents=True)

        svc, library = _make_service(tmp_path, f'''\
sources:
  shared:
    path: {shared_dir}
  product:
    path: {product_dir}
    depends_on: [shared]
  extension:
    path: {extension_dir}
    depends_on: [product]
''')
        # Coverage walks the source path and treats any file that
        # appears in some doc's source_files as documented. If we
        # leak extension docs into product's coverage view, an
        # extension doc whose source_files reference tracked.py
        # would incorrectly mark it as documented.
        library.add_document(
            content_type='explanation', title='product-doc',
            content='product content',
            source_name='product',
            # Note: source_files stores paths; coverage matches by
            # exact string. We don't add tracked.py here — so without
            # any docs, coverage should report 0 documented files.
            source_files=['product/something-else.py'],
        )
        library.add_document(
            content_type='explanation', title='extension-doc',
            content='extension content',
            source_name='extension',
            # Extension claims to document product/tracked.py. If the
            # closure leak isn't fixed, coverage(source='product') will
            # see this and count tracked.py as documented — even though
            # the documenting doc belongs to a different source.
            source_files=[str(product_dir / 'tracked.py')],
        )

        response = svc.coverage(source='product')

        # tracked.py and untracked.py both exist; neither should be
        # claimed-as-documented by a product-side doc (only the
        # extension claim points at tracked.py, and that's out of
        # closure). So coverage_percent should be 0.
        assert response.documented_count == 0
        assert response.undocumented_count == 2

        # T1 still holds: list_all(source='product') scopes correctly.
        list_response = svc.list_all(source='product')
        titles = {d.title for d in list_response.documents}
        assert titles == {'product-doc'}
        assert 'extension-doc' not in titles

    # ---- T5 -----------------------------------------------------------
    # CLI dispatch follows the same resolve-and-wrap discipline. A CLI
    # command given ``--source product`` builds a ScopedLibrary the
    # same way the MCP service does; without ``--source`` the resolver
    # falls back to cwd auto-detection; with neither, fail-closed.
    #
    # Implementation-wise this is enforced by the shared
    # ``scope_resolution.make_scoped_library`` helper — the CLI calls
    # the same function the MCP service does, so the contracts can't
    # drift apart. The test here verifies the helper's contract holds
    # through the CLI surface end-to-end.
    def test_t5_cli_uses_shared_resolver(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        import pytest
        from scope_resolution import make_scoped_library
        from library import Library

        product_dir = tmp_path / 'src' / 'product'
        shared_dir = tmp_path / 'src' / 'shared'
        extension_dir = tmp_path / 'src' / 'extension'
        for d in (shared_dir, product_dir, extension_dir):
            d.mkdir(parents=True)

        cfg_path = _write_config(tmp_path, f'''\
sources:
  shared:
    path: {shared_dir}
  product:
    path: {product_dir}
    depends_on: [shared]
  extension:
    path: {extension_dir}
    depends_on: [product]
''')
        from config import Config
        cfg = Config(cfg_path)

        with Library(tmp_path / 'library.db') as library:
            library.add_document(
                content_type='explanation', title='product-doc',
                content='product content', source_name='product',
                source_files=['product/foo.py'],
            )
            library.add_document(
                content_type='explanation', title='extension-doc',
                content='extension content', source_name='extension',
                source_files=['extension/foo.py'],
            )

            # Explicit --source argument.
            scoped = make_scoped_library(cfg, library, 'product')
            hits = scoped.find_documents_by_source_files(['foo.py'])
            assert {d.title for d in hits} == {'product-doc'}

            # No --source argument; cwd inside product_dir.
            monkeypatch.chdir(product_dir)
            scoped_cwd = make_scoped_library(cfg, library, None)
            hits_cwd = scoped_cwd.find_documents_by_source_files(['foo.py'])
            assert {d.title for d in hits_cwd} == {'product-doc'}

            # No --source, cwd outside any source, no default_source.
            unrelated = tmp_path / 'unrelated'
            unrelated.mkdir()
            monkeypatch.chdir(unrelated)
            with pytest.raises(LookupError) as exc:
                make_scoped_library(cfg, library, None)
            assert 'no source' in str(exc.value).lower()

    # ---- T6 -----------------------------------------------------------
    # The directional flip flows through the request boundary without
    # any per-source branch in the handler or the helper. A leaf
    # source's wrapper has the reverse closure baked in by
    # ``Config.scope_closure``; the handler just consumes it.
    #
    # This test verifies the end-to-end semantics: ``source='shared'``
    # builds a wrapper with closure ``{shared, product, extension}``,
    # so docs from product AND extension surface when the leaf-source
    # caller asks for documents. Symmetrically, a non-leaf source
    # (``source='product'``) produces only ``{product, shared}``.
    def test_t6_leaf_source_sees_reverse_closure(
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
            content_type='explanation', title='shared-doc',
            content='shared content', source_name='shared',
        )
        library.add_document(
            content_type='explanation', title='product-doc',
            content='product content', source_name='product',
        )
        library.add_document(
            content_type='explanation', title='extension-doc',
            content='extension content', source_name='extension',
        )

        # Leaf source: reverse closure surfaces the whole consumer set.
        leaf_response = svc.list_all(source='shared')
        leaf_titles = {d.title for d in leaf_response.documents}
        assert leaf_titles == {
            'shared-doc', 'product-doc', 'extension-doc',
        }

        # Non-leaf source: forward closure only.
        mid_response = svc.list_all(source='product')
        mid_titles = {d.title for d in mid_response.documents}
        assert mid_titles == {'shared-doc', 'product-doc'}
        assert 'extension-doc' not in mid_titles

        # Top of the chain: also forward only.
        top_response = svc.list_all(source='extension')
        top_titles = {d.title for d in top_response.documents}
        assert top_titles == {
            'shared-doc', 'product-doc', 'extension-doc',
        }

    # ---- T7 (deferred) ------------------------------------------------
    # Plan T7 — "every entry point migrated" plus the rename of public
    # Library data methods to ``_*_unscoped`` — is deferred to Phase 5
    # where the lint check provides the structural enforcement. The
    # rename would be a mechanical sweep across ~60 call sites in
    # production code and a similar count in tests; doing it inside an
    # evolutionary-TDD cycle would leave the repo red at every
    # intermediate commit. The shared helper and the per-handler
    # migration pattern that T1-T6 set up is what every remaining
    # caller will use; Phase 5 then catches anything that didn't.
