"""Pytest wrapper for the Node-level Vue SFC extractor regression test.

The extractor must speak both @vue/compiler-sfc dialects (Vue 2.7's
``parse({source, filename}) -> descriptor`` and Vue 3's
``parse(source, {filename}) -> {descriptor, errors}``). The hermetic
assertions live in ``scripts/scip/extract-vue-scripts.test.js`` (fake
``sfc`` stubs, no real Vue install); this just runs them under node so
they're part of the suite.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_TEST_JS = (
    Path(__file__).resolve().parent.parent
    / 'scripts' / 'scip' / 'extract-vue-scripts.test.js'
)


@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
def test_vue_extractor_supports_vue2_and_vue3():
    assert _TEST_JS.exists(), f'missing {_TEST_JS}'
    result = subprocess.run(
        ['node', str(_TEST_JS)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f'node test failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}'
    )
