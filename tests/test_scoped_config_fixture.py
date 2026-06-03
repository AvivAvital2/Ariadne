"""Pins behavior of the shared ``install_test_config`` helper.

The earlier version wrote to ``tmp_path / '..'`` — a session-shared
parent directory that two parallel pytest workers with the same source
name would race to overwrite. The new contract: each call produces a
process-unique cfg directory that no other test in the run shares,
AND the cfg path lives outside any plausible source_root so the
catalog walk doesn't pick it up.
"""
from __future__ import annotations

from pathlib import Path


class TestInstallTestConfig:
    def test_distinct_calls_produce_distinct_cfg_paths(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Two install_test_config invocations with different tmp_paths
        must NOT write to the same yaml location — otherwise
        pytest-xdist workers racing on the same source name would
        clobber each other's content. Reads ``config.get_config`` via
        module attribute (matches production's late-import pattern).
        """
        import config
        from tests._scoped_config_fixture import install_test_config

        install_test_config(monkeypatch, tmp_path, 'src')
        first_path = Path(config.get_config().config_path).resolve()

        other_tmp = tmp_path.parent / (tmp_path.name + '-worker2')
        other_tmp.mkdir()
        install_test_config(monkeypatch, other_tmp, 'src')
        second_path = Path(config.get_config().config_path).resolve()

        assert first_path != second_path, (
            'install_test_config produced colliding cfg paths for '
            f'two distinct tmp_paths; both wrote to {first_path}'
        )

    def test_cfg_path_is_not_inside_tmp_path(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The yaml must live OUTSIDE the test's tmp_path because
        ``sync_source_catalog``'s default ``exclude_dir_names=()``
        doesn't apply DEFAULT_EXCLUDE_POLICY — so a yaml anywhere
        under tmp_path would leak into catalog-file iteration in
        tests that assert a specific file count.
        """
        import config
        from tests._scoped_config_fixture import install_test_config

        install_test_config(monkeypatch, tmp_path, 'src')
        cfg_real = Path(config.get_config().config_path).resolve()
        tmp_real = tmp_path.resolve()
        assert tmp_real not in cfg_real.parents, (
            f'config yaml at {cfg_real} is inside tmp_path '
            f'{tmp_real}; catalog walks under tmp_path will see it '
            'and break tests that count expected catalog files'
        )
