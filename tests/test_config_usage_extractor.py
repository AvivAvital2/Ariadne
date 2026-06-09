"""extract_config_reads — Tier 2 Feature 3 (config-getter call-site enumerator).

Walks the populated ``string_literals`` candidate set, classifies each
candidate line with ``inspect_definition_rhs`` (chain-aware, Feature 2),
resolves the key against ``config_values``, and emits one
``ConfigRead(file, line, col, key, value|None, confidence)`` per confirmed
getter call. Falls back to a Tier-1 string-match for files whose language
the inspector can't read.

Synthetic, repo/language-agnostic fixtures only — never real source/keys.
See designs/config-code-bridge/tier2-resolution.md (Feature 3).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def conn():
    from library.scip import init_scip_schema

    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    yield c
    c.close()


def _add_literals(conn, source, rows) -> None:
    """rows: list of (file, line, col, value)."""
    from docgen.scip_string_literal_index import (
        StringLiteral,
        persist_string_literals,
    )

    persist_string_literals(
        source_name=source,
        literals=[
            StringLiteral(
                file=Path(f), line_start=ln, col_start=col,
                value=v, owning_symbol_id=None,
            )
            for (f, ln, col, v) in rows
        ],
        conn=conn,
    )


def _add_config_values(conn, source, rows) -> None:
    """rows: list of (file, key, value, line)."""
    from docgen.scip_config_index import persist_config_values
    from docgen.scip_config_scanners import ConfigValue

    persist_config_values(
        source_name=source,
        config_values=[
            ConfigValue(file=Path(f), key=k, value=v, line_start=ln)
            for (f, k, v, ln) in rows
        ],
        conn=conn,
    )


def test_simple_getter_resolves_value(conn, tmp_path) -> None:
    from docgen.scip_config_usage_extractor import extract_config_reads

    reader = tmp_path / 'reader.scala'
    reader.write_text(
        'val ttl = cfg.getString("svc.cache.ttl")\n', encoding='utf-8',
    )
    _add_literals(conn, 'src1', [(str(reader), 1, 24, 'svc.cache.ttl')])
    _add_config_values(conn, 'src1', [('app.conf', 'svc.cache.ttl', '30', 5)])

    reads = extract_config_reads(source_name='src1', conn=conn)
    assert len(reads) == 1
    assert reads[0].key == 'svc.cache.ttl'
    assert reads[0].value == '30'
    assert reads[0].confidence == 'config-resolved'
    assert reads[0].line == 1


def test_split_path_chain_is_single_read(conn, tmp_path) -> None:
    from docgen.scip_config_usage_extractor import extract_config_reads

    reader = tmp_path / 'reader.scala'
    reader.write_text(
        'val flag = cfg.getConfig("grp").getBoolean("enabled")\n',
        encoding='utf-8',
    )
    _add_literals(conn, 'src1', [
        (str(reader), 1, 25, 'grp'),
        (str(reader), 1, 44, 'enabled'),
    ])
    _add_config_values(conn, 'src1', [('app.conf', 'grp.enabled', 'true', 6)])

    reads = extract_config_reads(source_name='src1', conn=conn)
    assert len(reads) == 1
    assert reads[0].key == 'grp.enabled'
    assert reads[0].value == 'true'


def test_undeclared_key_resolves_to_none(conn, tmp_path) -> None:
    from docgen.scip_config_usage_extractor import extract_config_reads

    reader = tmp_path / 'reader.scala'
    reader.write_text(
        'val size = cfg.getInt("svc.cache.size")\n', encoding='utf-8',
    )
    _add_literals(conn, 'src1', [(str(reader), 1, 22, 'svc.cache.size')])
    # config_values has no entry for the key -> value is None, read still real.

    reads = extract_config_reads(source_name='src1', conn=conn)
    assert len(reads) == 1
    assert reads[0].value is None
    assert reads[0].confidence == 'config-resolved'


def test_non_getter_literal_excluded(conn, tmp_path) -> None:
    from docgen.scip_config_usage_extractor import extract_config_reads

    reader = tmp_path / 'reader.scala'
    reader.write_text('log.info("svc.cache.ttl")\n', encoding='utf-8')
    _add_literals(conn, 'src1', [(str(reader), 1, 9, 'svc.cache.ttl')])
    _add_config_values(conn, 'src1', [('app.conf', 'svc.cache.ttl', '30', 5)])

    assert extract_config_reads(source_name='src1', conn=conn) == []


def test_unsupported_language_string_match_fallback(conn, tmp_path) -> None:
    from docgen.scip_config_usage_extractor import extract_config_reads

    ruby = tmp_path / 'reader.rb'  # extension the inspector doesn't handle
    ruby.write_text('x = settings["svc.cache.ttl"]\n', encoding='utf-8')
    _add_literals(conn, 'src1', [
        (str(ruby), 1, 13, 'svc.cache.ttl'),
        (str(ruby), 1, 40, 'unrelated.literal'),  # matches no key -> dropped
    ])
    _add_config_values(conn, 'src1', [('app.conf', 'svc.cache.ttl', '30', 5)])

    reads = extract_config_reads(source_name='src1', conn=conn)
    assert len(reads) == 1
    assert reads[0].key == 'svc.cache.ttl'
    assert reads[0].confidence == 'string-match'
    assert reads[0].value == '30'


def test_unreadable_source_string_match_fallback(conn, tmp_path) -> None:
    from docgen.scip_config_usage_extractor import extract_config_reads

    ghost = tmp_path / 'ghost.scala'  # supported ext, never written -> OSError
    _add_literals(conn, 'src1', [(str(ghost), 1, 10, 'svc.cache.ttl')])
    _add_config_values(conn, 'src1', [('app.conf', 'svc.cache.ttl', '30', 5)])

    reads = extract_config_reads(source_name='src1', conn=conn)
    assert len(reads) == 1
    assert reads[0].confidence == 'string-match'
    assert reads[0].value == '30'


def test_each_file_parsed_once(conn, tmp_path, monkeypatch) -> None:
    """The extractor parses each source file ONCE, not once per getter
    line — the linear-time contract. Regression guard against the old
    per-line re-parse (which was O(lines x filesize) per file)."""
    import docgen.scip_definition_inspector as insp
    from docgen.scip_config_usage_extractor import extract_config_reads

    reader = tmp_path / 'reader.scala'
    n = 12
    reader.write_text(
        '\n'.join(f'val v{i} = cfg.getString("k.{i}")' for i in range(n)) + '\n',
        encoding='utf-8',
    )
    _add_literals(conn, 'src1', [(str(reader), i + 1, 24, f'k.{i}') for i in range(n)])
    _add_config_values(conn, 'src1', [('app.conf', f'k.{i}', str(i), 1) for i in range(n)])

    calls = {'n': 0}
    real = insp.SgRoot

    def counting(text, lang):
        calls['n'] += 1
        return real(text, lang)

    monkeypatch.setattr(insp, 'SgRoot', counting)

    reads = extract_config_reads(source_name='src1', conn=conn)
    assert len(reads) == n      # every getter found
    assert calls['n'] == 1      # ONE parse for the whole file, not n


def _cols_for(text, pairs):
    """(file, line, col, value) rows with col = the value's real column on
    its line, so two literals on one chain line get distinct positions."""
    lines = text.split('\n')
    return [
        (None, ln, lines[ln - 1].index(val), val) for (ln, val) in pairs
    ]


def test_scala_differential_exact_read_set(conn, tmp_path) -> None:
    """End-to-end: one Scala file mixing every getter/non-getter shape;
    assert the EXACT ConfigRead set. Pins functional behavior at the
    integration level — any false positive (a literal/log/non-getter that
    happens to match a declared key) or false negative (a missed getter,
    chain, bare/opaque receiver, or dynamic arg) fails here."""
    from docgen.scip_config_usage_extractor import extract_config_reads

    reader = tmp_path / 'reader.scala'
    src = (
        'val a = cfg.getString("simple.key")\n'              # 1 simple getter
        'val b = cfg.getConfig("grp").getBoolean("flag")\n'  # 2 split-path chain
        'val c = cfg.getConfig("a").getConfig("b").getInt("deep")\n'  # 3 nested chain
        'val d = cfg.getInt("absent.key")\n'                 # 4 getter, key undeclared
        'val e = getString("bare.key")\n'                    # 5 bare-name getter
        'val f = sub.getString("opaque.key")\n'              # 6 opaque receiver
        'val g = cfg.lookup("ns").getString("via.nongetconfig")\n'  # 7 non-getConfig in chain
        'val h = cfg.getConfig(dyn).getString("dyn.parent")\n'  # 8 dynamic getConfig arg
        'val i = "looks.like.key"\n'                         # 9 plain literal (NOT a read)
        'log.info("simple.key")\n'                           # 10 log call (NOT a read)
        'val k = compute("another.literal")\n'               # 11 non-getter call (NOT a read)
        'val l = cfg.getString("simple.key")\n'              # 12 second read of simple.key
    )
    reader.write_text(src, encoding='utf-8')

    literal_lines = [
        (1, 'simple.key'),
        (2, 'grp'), (2, 'flag'),
        (3, 'a'), (3, 'b'), (3, 'deep'),
        (4, 'absent.key'),
        (5, 'bare.key'),
        (6, 'opaque.key'),
        (7, 'ns'), (7, 'via.nongetconfig'),
        (8, 'dyn.parent'),
        (9, 'looks.like.key'),
        (10, 'simple.key'),
        (11, 'another.literal'),
        (12, 'simple.key'),
    ]
    _add_literals(conn, 'src1', [
        (str(reader), ln, col, val)
        for (_f, ln, col, val) in _cols_for(src, literal_lines)
    ])
    # Declare keys — crucially including the ones used by the non-getter
    # lines (looks.like.key / another.literal / simple.key-in-the-log), so
    # a string-match would wrongly include them; config-resolved must not.
    _add_config_values(conn, 'src1', [
        ('app.conf', 'simple.key', 'S', 1),
        ('app.conf', 'grp.flag', 'true', 2),
        ('app.conf', 'a.b.deep', '42', 3),
        ('app.conf', 'opaque.key', 'O', 4),
        ('app.conf', 'via.nongetconfig', 'V', 5),
        ('app.conf', 'dyn.parent', 'D', 6),
        ('app.conf', 'looks.like.key', 'L', 7),     # declared but only ever a literal
        ('app.conf', 'another.literal', 'A', 8),    # declared but only a compute() arg
        # absent.key and bare.key intentionally NOT declared -> value None
    ])

    reads = extract_config_reads(source_name='src1', conn=conn)
    got = {(r.line, r.key, r.value, r.confidence) for r in reads}
    expected = {
        (1, 'simple.key', 'S', 'config-resolved'),
        (2, 'grp.flag', 'true', 'config-resolved'),       # one read for the chain, not two
        (3, 'a.b.deep', '42', 'config-resolved'),
        (4, 'absent.key', None, 'config-resolved'),       # read real, value undeclared
        (5, 'bare.key', None, 'config-resolved'),
        (6, 'opaque.key', 'O', 'config-resolved'),
        (7, 'via.nongetconfig', 'V', 'config-resolved'),  # lookup() adds no segment
        (8, 'dyn.parent', 'D', 'config-resolved'),        # dynamic getConfig arg dropped
        (12, 'simple.key', 'S', 'config-resolved'),       # same key, distinct site
    }
    assert got == expected
    assert len(reads) == len(expected)  # no duplicates (e.g. chain double-count)


def test_js_differential_exact_read_set(conn, tmp_path) -> None:
    """Same exact-set rigor for the JavaScript classification path
    (member-call getter, getConfig chain, subscript getter; non-getter
    call and plain literal excluded though their text is a declared key)."""
    from docgen.scip_config_usage_extractor import extract_config_reads

    reader = tmp_path / 'reader.js'
    src = (
        "const a = cfg.getString('js.simple');\n"            # 1 getter
        "const b = cfg.getConfig('jsg').getBoolean('jsflag');\n"  # 2 chain
        "const c = config['js.subscript'];\n"                # 3 subscript getter
        "const d = compute('js.literal');\n"                 # 4 non-getter (NOT a read)
        "const e = 'js.plain';\n"                            # 5 plain literal (NOT a read)
    )
    reader.write_text(src, encoding='utf-8')

    literal_lines = [
        (1, 'js.simple'),
        (2, 'jsg'), (2, 'jsflag'),
        (3, 'js.subscript'),
        (4, 'js.literal'),
        (5, 'js.plain'),
    ]
    _add_literals(conn, 'src1', [
        (str(reader), ln, col, val)
        for (_f, ln, col, val) in _cols_for(src, literal_lines)
    ])
    _add_config_values(conn, 'src1', [
        ('app.conf', 'js.simple', 'JS', 1),
        ('app.conf', 'jsg.jsflag', 'JF', 2),
        ('app.conf', 'js.subscript', 'SUB', 3),
        ('app.conf', 'js.literal', 'LIT', 4),   # declared but only a compute() arg
        ('app.conf', 'js.plain', 'PLN', 5),     # declared but only a literal
    ])

    reads = extract_config_reads(source_name='src1', conn=conn)
    got = {(r.line, r.key, r.value, r.confidence) for r in reads}
    expected = {
        (1, 'js.simple', 'JS', 'config-resolved'),
        (2, 'jsg.jsflag', 'JF', 'config-resolved'),
        (3, 'js.subscript', 'SUB', 'config-resolved'),
    }
    assert got == expected
    assert len(reads) == len(expected)
