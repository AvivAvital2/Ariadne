"""Tests for the ariadne themes CLI surface (Themes plan, Phase 7).

Covers:
- Argparse wiring for nested `themes build|list|show <id>` subcommands.
- cmd_themes_build dispatches to docgen.themes.refresh_themes.
- cmd_themes_list / cmd_themes_show delegate to the themes_action helper
  (so the human-readable CLI output stays consistent with the MCP shape).
- 'themes' registered in cli_generation.HANDLERS.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from library import Library


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0 else arr


def _add_doc(
    library: Library,
    doc_id: str,
    *,
    content_type: str = 'theme',
    title: str | None = None,
    content: str = 'content',
) -> None:
    library.add_document(
        content_type=content_type,  # type: ignore[arg-type]
        title=title or f'doc {doc_id}',
        content=content,
        source_files=[],
        embedding=_unit([1.0, 0, 0, 0, 0, 0, 0, 0]),
        metadata={},
        doc_id=doc_id,
    )


def _bootstrap_theme(
    library: Library,
    cluster_id: str,
    *,
    members: list[str],
    title: str = 'Theme Title',
    content: str = '# Theme Title\n\nbody',
    coherent: bool = True,
) -> None:
    """Insert a placeholder theme + members for tests."""
    doc_id = f'theme-doc-{cluster_id}'
    _add_doc(library, doc_id, content_type='theme', title=title, content=content)
    library.add_theme(
        cluster_id=cluster_id,
        doc_id=doc_id,
        member_count=len(members),
        resolution=1.0,
        summary_hash='h',
        coherent=coherent,
        dirty=False,
    )
    for mid in members:
        _add_doc(library, mid, content_type='catalog', title=mid)
    library.set_theme_members(cluster_id, [(mid, 1.0) for mid in members])


@pytest.fixture
def library(tmp_path: Path):
    lib = Library(tmp_path / 'themes-cli-test.db')
    yield lib
    lib.close()


@pytest.fixture
def patched_get_library(monkeypatch, library: Library):
    """Make cli_generation.get_library / get_config return our library."""
    monkeypatch.setattr('cli.themes_cmd.get_library', lambda *_a, **_k: library)
    return library


# ---------------------------------------------------------------------------
# Argparse registration
# ---------------------------------------------------------------------------


class TestThemesArgparse:
    def test_themes_command_registered(self) -> None:
        """`ariadne themes` must be a valid command with three subcommands."""
        from cli.main import create_parser

        parser = create_parser()
        # The build subcommand must parse (no errors).
        args = parser.parse_args(['themes', 'build'])
        assert args.command == 'themes'
        assert getattr(args, 'themes_action', None) == 'build'

    def test_themes_list_subcommand_parses(self) -> None:
        from cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(['themes', 'list'])
        assert args.command == 'themes'
        assert args.themes_action == 'list'

    def test_themes_show_requires_cluster_id(self) -> None:
        from cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(['themes', 'show', 'abc-123'])
        assert args.command == 'themes'
        assert args.themes_action == 'show'
        assert args.cluster_id == 'abc-123'

    def test_themes_show_without_id_errors(self) -> None:
        from cli.main import create_parser

        parser = create_parser()
        # argparse exits with SystemExit when a required positional is missing.
        with pytest.raises(SystemExit):
            parser.parse_args(['themes', 'show'])

    def test_themes_handler_in_handlers_map(self) -> None:
        from cli.themes_cmd import HANDLERS

        assert 'themes' in HANDLERS


# ---------------------------------------------------------------------------
# cmd_themes_build
# ---------------------------------------------------------------------------


class TestThemesBuild:
    def test_build_invokes_refresh_themes(
        self, patched_get_library: Library, monkeypatch,
    ) -> None:
        """`ariadne themes build` must call refresh_themes — the actual
        clustering pipeline. Without this, 'build' is a hollow no-op.
        """
        import argparse

        from cli.themes_cmd import cmd_themes

        called: list[dict] = []

        async def fake_refresh(library, writer, **kwargs):
            called.append({'library': library, 'kwargs': kwargs})
            return {
                'path': 'noop',
                'changed': 0,
                'recluster_full': False,
                'summarized': 0,
                'incoherent': 0,
                'failed': 0,
                'total_dirty': 0,
            }

        monkeypatch.setattr('docgen.themes.refresh_themes', fake_refresh)

        args = argparse.Namespace(
            themes_action='build',
            db=None,
            cluster_id=None,
            coherent_only=True,
            source=None,
            limit=50,
        )
        rc = cmd_themes(args)
        assert rc == 0
        assert len(called) == 1, 'refresh_themes must run exactly once'

    @pytest.mark.asyncio
    async def test_build_phase_composes_into_running_event_loop(
        self, patched_get_library: Library, monkeypatch,
    ) -> None:
        """The themes-build phase must be awaitable so it composes into the
        already-running event loop of `ariadne onboard` (whose pipeline is
        async and awaits each phase). A synchronous ``asyncio.run()`` inside
        the phase raises 'asyncio.run() cannot be called from a running event
        loop' and leaves the build coroutine un-awaited. Regression for the
        onboard 'Building themes' phase failure.
        """
        import argparse

        from cli.themes_cmd import cmd_themes_build

        called: list[dict] = []

        async def fake_refresh(library, writer, **kwargs):
            called.append({'kwargs': kwargs})
            return {
                'path': 'noop', 'changed': 0, 'recluster_full': False,
                'summarized': 0, 'incoherent': 0, 'failed': 0,
                'total_dirty': 0,
            }

        monkeypatch.setattr('docgen.themes.refresh_themes', fake_refresh)

        args = argparse.Namespace(
            themes_action='build', db=None, cluster_id=None,
            coherent_only=True, source=None, limit=50,
        )
        # We are inside a running loop (pytest.mark.asyncio), exactly like
        # onboard's pipeline. Awaiting the build phase the way onboard does
        # must succeed — not raise, not return 1.
        rc = await cmd_themes_build(args)
        assert rc == 0
        assert len(called) == 1, 'refresh_themes must run exactly once'

    def test_build_returns_one_when_refresh_raises(
        self, patched_get_library: Library, monkeypatch,
    ) -> None:
        """A clustering crash should be reported with non-zero exit, not a
        silent success.
        """
        import argparse

        from cli.themes_cmd import cmd_themes

        async def boom(library, writer, **kwargs):
            raise RuntimeError('kaboom')

        monkeypatch.setattr('docgen.themes.refresh_themes', boom)

        args = argparse.Namespace(
            themes_action='build',
            db=None,
            cluster_id=None,
            coherent_only=True,
            source=None,
            limit=50,
        )
        rc = cmd_themes(args)
        assert rc == 1


# ---------------------------------------------------------------------------
# cmd_themes_list
# ---------------------------------------------------------------------------


class TestThemesList:
    def test_list_returns_zero_with_no_themes(
        self, patched_get_library: Library,
    ) -> None:
        import argparse

        from cli.themes_cmd import cmd_themes

        args = argparse.Namespace(
            themes_action='list',
            db=None,
            cluster_id=None,
            coherent_only=True,
            source=None,
            limit=50,
        )
        rc = cmd_themes(args)
        assert rc == 0

    def test_list_returns_zero_with_themes(
        self, patched_get_library: Library, capsys,
    ) -> None:
        import argparse

        from cli.themes_cmd import cmd_themes

        _bootstrap_theme(
            patched_get_library, 'c1', members=['el1', 'el2'], title='Retry Logic',
        )

        args = argparse.Namespace(
            themes_action='list',
            db=None,
            cluster_id=None,
            coherent_only=True,
            source=None,
            limit=50,
        )
        rc = cmd_themes(args)
        assert rc == 0
        out = capsys.readouterr().out
        # Cluster_id and title should both be visible to the user.
        assert 'c1' in out
        assert 'Retry' in out

    def test_list_excludes_incoherent_by_default(
        self, patched_get_library: Library, capsys,
    ) -> None:
        """Default behavior matches plan §5.6: coherent_only=True."""
        import argparse

        from cli.themes_cmd import cmd_themes

        _bootstrap_theme(
            patched_get_library, 'c1', members=['el1'],
            title='Coherent Retry', coherent=True,
        )
        _bootstrap_theme(
            patched_get_library, 'c2', members=['el2'],
            title='Noisy Cluster', coherent=False,
        )

        args = argparse.Namespace(
            themes_action='list',
            db=None,
            cluster_id=None,
            coherent_only=True,
            source=None,
            limit=50,
        )
        rc = cmd_themes(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert 'Coherent Retry' in out or 'c1' in out
        assert 'Noisy Cluster' not in out


# ---------------------------------------------------------------------------
# cmd_themes_show
# ---------------------------------------------------------------------------


class TestThemesShow:
    def test_show_known_cluster_returns_zero_and_prints_content(
        self, patched_get_library: Library, capsys,
    ) -> None:
        import argparse

        from cli.themes_cmd import cmd_themes

        _bootstrap_theme(
            patched_get_library, 'c1', members=['el1'],
            title='Configuration Loading',
            content='# Configuration Loading\n\nfull body text\n',
        )

        args = argparse.Namespace(
            themes_action='show',
            db=None,
            cluster_id='c1',
            coherent_only=True,
            source=None,
            limit=50,
        )
        rc = cmd_themes(args)
        assert rc == 0
        out = capsys.readouterr().out
        # The whole doc body must be printed (not just the title).
        assert 'full body text' in out

    def test_show_unknown_cluster_returns_one(
        self, patched_get_library: Library,
    ) -> None:
        """Unknown cluster_id is a user-facing error → non-zero exit."""
        import argparse

        from cli.themes_cmd import cmd_themes

        args = argparse.Namespace(
            themes_action='show',
            db=None,
            cluster_id='missing',
            coherent_only=True,
            source=None,
            limit=50,
        )
        rc = cmd_themes(args)
        assert rc == 1


# ---------------------------------------------------------------------------
# Dispatcher robustness
# ---------------------------------------------------------------------------


class TestThemesDispatcher:
    def test_unknown_themes_action_returns_one(
        self, patched_get_library: Library,
    ) -> None:
        import argparse

        from cli.themes_cmd import cmd_themes

        args = argparse.Namespace(
            themes_action=None,
            db=None,
            cluster_id=None,
            coherent_only=True,
            source=None,
            limit=50,
        )
        rc = cmd_themes(args)
        # Either prints help and returns 0, or returns non-zero — the contract
        # is that we don't crash. We accept both.
        assert rc in (0, 1)
