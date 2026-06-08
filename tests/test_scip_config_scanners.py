"""Contract for the config-source scanner registry (Phase 2o / Layer B).

Layer B's job is to extract ``(file, key, value, line)`` tuples from
configuration files in the source tree. The output feeds Layer C
(Phase 2q config-value index) and is later consumed by the resolution
traversal engine (Phase 2s) when it walks SCIP refs from a sink-call
site backward through config-key lookups.

Layer B does NOT directly produce ``(code_dir → env_dir)`` mappings.
That requires SCIP-ref traversal (Scala calls X, X resolves to a HOCON
key, that key has a value pointing at the env). Layer B just provides
the config-side raw material.

MVP scope (this slice):

- HOCON (``*.conf``) — uses the in-tree Lark parser
- Dotenv (``.env``, ``.env.local``, ``.env.*``) — KEY=VALUE per line

Deferred to Phase 2o.b (when needed):

- Dockerfile (ENTRYPOINT/CMD/RUN/ENV)
- YAML (``*.yaml``/``*.yml`` for kustomize patches and similar)

These tests are RED until ``docgen/scip_config_scanners.py`` exists.
"""
from __future__ import annotations

from pathlib import Path


def _touch(path: Path, content: str = '') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


# ---------------------------------------------------------------------------
# ConfigValue value class
# ---------------------------------------------------------------------------


class TestConfigValueShape:
    def test_config_value_is_frozen(self, tmp_path: Path) -> None:
        """ConfigValue is an attrs @frozen value object so tests can
        compare by equality and callers can pass it around without
        defensive copies."""
        from docgen.scip_config_scanners import ConfigValue

        cv = ConfigValue(
            file=tmp_path / 'app.conf',
            key='resources.python',
            value='/usr/bin/python3',
            line_start=42,
        )
        # @frozen raises on attribute assignment
        try:
            cv.value = 'mutated'  # type: ignore[misc]
        except Exception:
            return
        raise AssertionError(
            'ConfigValue should be @frozen',
        )


# ---------------------------------------------------------------------------
# Dotenv scanner
# ---------------------------------------------------------------------------


class TestDotenvScanner:
    def test_simple_key_value_pairs(self, tmp_path: Path) -> None:
        """Each ``KEY=VALUE`` line yields one ConfigValue with the
        file's 1-indexed line_start."""
        from docgen.scip_config_scanners import scan_dotenv

        env_file = tmp_path / '.env'
        _touch(
            env_file,
            'PYTHON_PATH=/usr/local/bin/python3\n'
            'AZUREML_HOME=/opt/azureml\n',
        )

        values = scan_dotenv(env_file)
        assert len(values) == 2

        by_key = {v.key: v for v in values}
        assert by_key['PYTHON_PATH'].value == '/usr/local/bin/python3'
        assert by_key['PYTHON_PATH'].line_start == 1
        assert by_key['AZUREML_HOME'].value == '/opt/azureml'
        assert by_key['AZUREML_HOME'].line_start == 2

    def test_quoted_values_unquoted(self, tmp_path: Path) -> None:
        """Surrounding quotes (single or double) are stripped — they're
        shell-quote syntax, not part of the value."""
        from docgen.scip_config_scanners import scan_dotenv

        env_file = tmp_path / '.env'
        _touch(
            env_file,
            'DOUBLE="some value"\n'
            "SINGLE='other value'\n",
        )

        values = scan_dotenv(env_file)
        by_key = {v.key: v for v in values}
        assert by_key['DOUBLE'].value == 'some value'
        assert by_key['SINGLE'].value == 'other value'

    def test_comments_ignored(self, tmp_path: Path) -> None:
        """Lines starting with ``#`` are comments and don't yield a
        ConfigValue."""
        from docgen.scip_config_scanners import scan_dotenv

        env_file = tmp_path / '.env'
        _touch(
            env_file,
            '# This is a comment\n'
            'REAL_KEY=value\n'
            '   # indented comment\n',
        )

        values = scan_dotenv(env_file)
        assert len(values) == 1
        assert values[0].key == 'REAL_KEY'

    def test_blank_lines_ignored(self, tmp_path: Path) -> None:
        """Blank lines don't yield ConfigValues."""
        from docgen.scip_config_scanners import scan_dotenv

        env_file = tmp_path / '.env'
        _touch(env_file, 'A=1\n\n\nB=2\n')

        values = scan_dotenv(env_file)
        assert len(values) == 2
        # B is on line 4 (1-indexed), not line 2
        by_key = {v.key: v for v in values}
        assert by_key['A'].line_start == 1
        assert by_key['B'].line_start == 4

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        from docgen.scip_config_scanners import scan_dotenv

        env_file = tmp_path / '.env'
        _touch(env_file, '')
        assert scan_dotenv(env_file) == []

    def test_value_with_equals_sign_preserved(
        self, tmp_path: Path,
    ) -> None:
        """``KEY=foo=bar`` yields ``key='KEY', value='foo=bar'`` — only
        the FIRST ``=`` separates key from value."""
        from docgen.scip_config_scanners import scan_dotenv

        env_file = tmp_path / '.env'
        _touch(env_file, 'CONNECTION_STRING=user=admin;pass=secret\n')

        values = scan_dotenv(env_file)
        assert len(values) == 1
        assert values[0].value == 'user=admin;pass=secret'


# ---------------------------------------------------------------------------
# HOCON scanner
# ---------------------------------------------------------------------------


class TestHoconScanner:
    def test_simple_key_value_extracted(self, tmp_path: Path) -> None:
        """HOCON ``key = "value"`` yields a ConfigValue with the dotted
        key and stripped string value."""
        from docgen.scip_config_scanners import scan_hocon

        conf = tmp_path / 'app.conf'
        _touch(
            conf,
            'resources.python = "/usr/bin/python3"\n',
        )

        values = scan_hocon(conf)
        assert len(values) == 1
        v = values[0]
        assert v.key == 'resources.python'
        assert v.value == '/usr/bin/python3'
        assert v.file == conf
        assert v.line_start == 1

    def test_nested_block_flattened_to_dotted_key(
        self, tmp_path: Path,
    ) -> None:
        """HOCON nested blocks ``resources { python = "..." }`` flatten
        to dotted keys like ``resources.python``."""
        from docgen.scip_config_scanners import scan_hocon

        conf = tmp_path / 'app.conf'
        _touch(
            conf,
            'resources {\n'
            '  python = "/usr/bin/python3"\n'
            '  azureMLPythonCommand = "/opt/azureml/bin/python"\n'
            '}\n',
        )

        values = scan_hocon(conf)
        keys = {v.key for v in values}
        assert 'resources.python' in keys
        assert 'resources.azureMLPythonCommand' in keys

    def test_scalaproject_pattern_python_command(
        self, tmp_path: Path,
    ) -> None:
        """The exact pattern from scalaproject's platform.conf — a deeply
        nested key whose value points at a deployed Python interpreter."""
        from docgen.scip_config_scanners import scan_hocon

        conf = tmp_path / 'platform.conf'
        _touch(
            conf,
            'resources {\n'
            '  azureMLPythonCommand = "/opt/azureml/.venv/bin/python"\n'
            '}\n',
        )

        values = scan_hocon(conf)
        target = next(
            (v for v in values
             if v.key == 'resources.azureMLPythonCommand'),
            None,
        )
        assert target is not None
        assert target.value == '/opt/azureml/.venv/bin/python'

    def test_multiple_top_level_keys(self, tmp_path: Path) -> None:
        """Each top-level key in HOCON yields its own ConfigValue."""
        from docgen.scip_config_scanners import scan_hocon

        conf = tmp_path / 'app.conf'
        _touch(
            conf,
            'foo = "first"\n'
            'bar = "second"\n'
            'baz = "third"\n',
        )

        values = scan_hocon(conf)
        keys = sorted(v.key for v in values)
        assert keys == ['bar', 'baz', 'foo']

    def test_non_string_values_skipped_or_stringified(
        self, tmp_path: Path,
    ) -> None:
        """Numbers and booleans aren't useful as Python interpreter
        paths or config-key references; the scanner may either skip
        them or stringify them, but must not crash."""
        from docgen.scip_config_scanners import scan_hocon

        conf = tmp_path / 'app.conf'
        _touch(
            conf,
            'count = 42\n'
            'enabled = true\n'
            'name = "x"\n',
        )

        values = scan_hocon(conf)
        # name is definitely captured; count/enabled may or may not be
        keys = {v.key for v in values}
        assert 'name' in keys

    def test_value_concatenation_is_skipped_not_truncated(
        self, tmp_path: Path,
    ) -> None:
        """HOCON value concatenation (`"http://"${host}`,
        `${base.dir}/log/events`) can't be resolved to a single flat
        value here. The scanner must SKIP it, never store a truncated
        first-atom value (`http://`) that would mislead sink resolution."""
        from docgen.scip_config_scanners import scan_hocon

        conf = tmp_path / 'app.conf'
        _touch(
            conf,
            'plain = "kept"\n'
            'endpoint = "http://"${host}\n'        # string + substitution
            'logdir = ${base.dir}/log/events\n',   # substitution + path
        )

        values = scan_hocon(conf)
        by_key = {v.key: v.value for v in values}
        assert by_key.get('plain') == 'kept'
        # Neither concatenation is stored — and crucially not as a truncated
        # partial value.
        assert 'endpoint' not in by_key, (
            f"concatenation stored a partial value: {by_key.get('endpoint')!r}"
        )
        assert 'logdir' not in by_key


# ---------------------------------------------------------------------------
# Aggregator — walks source tree and dispatches by file type
# ---------------------------------------------------------------------------


class TestScanConfigSources:
    def test_walks_subdirectories(self, tmp_path: Path) -> None:
        """The aggregator finds config files anywhere under source_root,
        not just at the top level."""
        from docgen.scip_config_scanners import scan_config_sources

        _touch(
            tmp_path / 'web' / 'conf' / 'master' / 'platform.conf',
            'resources.python = "/usr/bin/python3"\n',
        )
        _touch(
            tmp_path / '.env',
            'API_KEY=secret\n',
        )

        values = scan_config_sources(tmp_path)
        assert len(values) >= 2
        keys = {v.key for v in values}
        assert 'resources.python' in keys
        assert 'API_KEY' in keys

    def test_dispatches_dotenv_to_dotenv_scanner(
        self, tmp_path: Path,
    ) -> None:
        """``.env`` files use the dotenv scanner regardless of where
        they live in the tree."""
        from docgen.scip_config_scanners import scan_config_sources

        _touch(
            tmp_path / 'services' / 'foo' / '.env',
            'PYTHON_PATH=/usr/bin/python\n',
        )

        values = scan_config_sources(tmp_path)
        py_path_values = [v for v in values if v.key == 'PYTHON_PATH']
        assert len(py_path_values) == 1
        assert py_path_values[0].value == '/usr/bin/python'

    def test_dispatches_conf_to_hocon_scanner(
        self, tmp_path: Path,
    ) -> None:
        """``.conf`` files use the HOCON scanner."""
        from docgen.scip_config_scanners import scan_config_sources

        _touch(
            tmp_path / 'config' / 'app.conf',
            'name = "myapp"\n',
        )

        values = scan_config_sources(tmp_path)
        name_values = [v for v in values if v.key == 'name']
        assert len(name_values) == 1
        assert name_values[0].value == 'myapp'

    def test_unsupported_file_extensions_ignored(
        self, tmp_path: Path,
    ) -> None:
        """Files we don't have a scanner for don't appear in the output
        and don't crash the aggregator."""
        from docgen.scip_config_scanners import scan_config_sources

        _touch(tmp_path / 'README.md', '# This is markdown\n')
        _touch(tmp_path / 'app.py', 'print("python source")\n')
        _touch(tmp_path / 'data.json', '{"key": "value"}\n')

        values = scan_config_sources(tmp_path)
        # No values from any of these
        files = {v.file for v in values}
        assert tmp_path / 'README.md' not in files
        assert tmp_path / 'app.py' not in files

    def test_excluded_dirs_skipped(self, tmp_path: Path) -> None:
        """``exclude_dirs`` (matching directory NAMES at any depth)
        skips entire subtrees during the walk."""
        from docgen.scip_config_scanners import scan_config_sources

        # Config in an excluded directory
        _touch(
            tmp_path / 'node_modules' / 'app.conf',
            'should_not_appear = "x"\n',
        )
        # Config in a normal directory
        _touch(
            tmp_path / 'real' / 'app.conf',
            'real_key = "y"\n',
        )

        values = scan_config_sources(
            tmp_path,
            exclude_dirs=frozenset({'node_modules'}),
        )

        keys = {v.key for v in values}
        assert 'should_not_appear' not in keys
        assert 'real_key' in keys

    def test_empty_tree_returns_empty_list(self, tmp_path: Path) -> None:
        """No config files anywhere → empty result, no errors."""
        from docgen.scip_config_scanners import scan_config_sources

        # Empty tree (just the directory itself)
        assert scan_config_sources(tmp_path) == []

    def test_file_paths_are_absolute(self, tmp_path: Path) -> None:
        """ConfigValue.file is the absolute path so callers don't need
        to know which source_root the scanner was called with."""
        from docgen.scip_config_scanners import scan_config_sources

        config = tmp_path / 'sub' / 'app.conf'
        _touch(config, 'key = "val"\n')

        values = scan_config_sources(tmp_path)
        assert len(values) == 1
        assert values[0].file.is_absolute()
        assert values[0].file == config.resolve()
