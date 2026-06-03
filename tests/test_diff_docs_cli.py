"""Tests for the ariadne diff-docs CLI (Catalog transition Phase 3.2).

`ariadne diff-docs --file <path>` runs both the legacy
(SourceAnalyzer + generate_for_module) and the new
(catalog_extractor + generate_from_elements) pipelines on a single Python
file and prints the prompts side-by-side. Use case: spot-checking during
the dual-run window.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, patch

from docgen.generator import DocGenerator


def _write(path: Path, src: str) -> None:
    path.write_text(dedent(src).lstrip('\n'), encoding='utf-8')


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


class TestArgparse:
    def test_diff_docs_command_registered(self) -> None:
        from cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(['diff-docs', '--file', 'x.py'])
        assert args.command == 'diff-docs'
        assert args.file == 'x.py'

    def test_diff_docs_handler_in_handlers_map(self) -> None:
        from cli.generation import HANDLERS

        assert 'diff-docs' in HANDLERS

    def test_diff_docs_supports_doc_type_filter(self) -> None:
        from cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(
            ['diff-docs', '--file', 'x.py', '--type', 'explanation'],
        )
        assert args.type == 'explanation'


# ---------------------------------------------------------------------------
# Behavior — runs both paths and prints both prompts
# ---------------------------------------------------------------------------


class TestRuns:
    def test_diff_docs_returns_zero_on_python_file(
        self, tmp_path: Path, capsys,
    ) -> None:
        import argparse

        f = tmp_path / 'm.py'
        _write(f, '''
            """alpha module."""
            class Greeter:
                def hello(self): return "hi"
            def public_fn(): return 1
        ''')

        from cli.generation import cmd_diff_docs

        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value='# stub',
        ):
            args = argparse.Namespace(
                file=str(f),
                type='explanation',
                with_llm=False,
                db=None,
            )
            rc = cmd_diff_docs(args)

        assert rc == 0
        out = capsys.readouterr().out
        # Both pipelines' output should be visible.
        assert 'Legacy' in out or 'legacy' in out
        assert 'Catalog' in out or 'catalog' in out or 'New' in out

    def test_diff_docs_shows_module_facts_from_both_paths(
        self, tmp_path: Path, capsys,
    ) -> None:
        """Both prompts should be visible side-by-side; the user should
        see the module name and function names mentioned in the output.
        """
        import argparse

        f = tmp_path / 'alpha.py'
        _write(f, '''
            """alpha doc."""
            def beta_function():
                return 0
        ''')

        from cli.generation import cmd_diff_docs

        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value='# stub',
        ):
            args = argparse.Namespace(
                file=str(f),
                type='explanation',
                with_llm=False,
                db=None,
            )
            rc = cmd_diff_docs(args)

        assert rc == 0
        out = capsys.readouterr().out
        # The function name should appear somewhere in the rendered output.
        assert 'beta_function' in out

    def test_diff_docs_returns_one_for_missing_file(
        self, tmp_path: Path,
    ) -> None:
        import argparse

        from cli.generation import cmd_diff_docs

        args = argparse.Namespace(
            file=str(tmp_path / 'does_not_exist.py'),
            type='explanation',
            with_llm=False,
            db=None,
        )
        rc = cmd_diff_docs(args)
        assert rc == 1

    def test_diff_docs_returns_one_for_non_python_file(
        self, tmp_path: Path,
    ) -> None:
        """The legacy path is Python-only — JS/JSON files have no
        meaningful comparison, so we exit early with a clear error.
        """
        import argparse

        f = tmp_path / 'config.json'
        f.write_text('{}', encoding='utf-8')

        from cli.generation import cmd_diff_docs

        args = argparse.Namespace(
            file=str(f),
            type='explanation',
            with_llm=False,
            db=None,
        )
        rc = cmd_diff_docs(args)
        assert rc == 1
