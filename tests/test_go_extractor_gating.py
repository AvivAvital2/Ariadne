"""The Go SCIP extractors must be gated on Go actually being present.

They load and parse the whole SCIP index for a source before filtering to
``.go`` documents, so running them on a non-Go corpus pays a full index load
(twice — routes + clients) for a guaranteed-empty result. ``_source_has_go``
is the cheap filesystem gate (mirrors how the Akka extractor gates on
``uses_akka_http``).
"""
from __future__ import annotations

from cli.index import _source_has_go


def test_source_has_go_true_for_go_source(tmp_path):
    (tmp_path / 'main.go').write_text('package main\n')
    assert _source_has_go(tmp_path) is True


def test_source_has_go_true_when_nested(tmp_path):
    (tmp_path / 'cmd').mkdir()
    (tmp_path / 'cmd' / 'server.go').write_text('package cmd\n')
    assert _source_has_go(tmp_path) is True


def test_source_has_go_false_without_go(tmp_path):
    (tmp_path / 'app.py').write_text('x = 1\n')
    sub = tmp_path / 'web'
    sub.mkdir()
    (sub / 'index.ts').write_text('export const x = 1\n')
    assert _source_has_go(tmp_path) is False
