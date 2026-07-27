"""Extractor tests: Lark parse tree → list[ElementInfo].

Layer 2 of the HOCON parser TDD. Layer 1 (test_hocon_grammar.py)
exercises the grammar in isolation; this layer exercises the bridge
that turns the parse tree into the ElementInfo shape the rest of the
catalog pipeline consumes.

Behavior contracts:

- `extract_elements` routes `.conf` files to the HOCON extractor — the
  return value is non-empty when the file has any keys.
- Each leaf key becomes a `hocon_key` ElementInfo.
- Each block (object value) ALSO becomes a `hocon_key` ElementInfo —
  so users searching for "activation" find the block, and "activation.pub"
  finds the leaf inside it.
- `parent_qualified_name` chains correctly: root entry -> module qn;
  nested entry -> parent block's full qn.
- `qualified_name` is the dotted path joined to the module qn,
  matching the json_key / yaml_key style.
- `line_start` / `line_end` come straight from Lark's position info.
- `signature` is a one-line summary suitable for catalog display
  (the line text, truncated to a reasonable length).
- `body_sha` differs between two structurally-identical files at
  different paths, so re-extraction can detect content changes.
"""
from __future__ import annotations

from pathlib import Path


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    return path


# ---------------------------------------------------------------------------
# Subtype: 'hocon_key' must be in the Literal so type checks pass and
# downstream filters (e.g. `subtype == 'hocon_key'`) work as strings.
# ---------------------------------------------------------------------------


def test_subtype_includes_hocon_key():
    from typing import get_args

    from docgen.catalog_extractor import Subtype

    assert 'hocon_key' in get_args(Subtype), (
        f"Subtype literal does not include 'hocon_key': {get_args(Subtype)}"
    )


# ---------------------------------------------------------------------------
# Single-key cases
# ---------------------------------------------------------------------------


def test_single_top_level_key(tmp_path: Path):
    from docgen.catalog_extractor import extract_elements

    f = _write(tmp_path, 'application.conf', 'foo = "bar"\n')
    elements = extract_elements(f, tmp_path)

    assert len(elements) == 1
    el = elements[0]
    assert el.subtype == 'hocon_key'
    assert el.language == 'hocon'
    assert el.qualified_name.endswith('.foo'), (
        f"qualified_name must end with key path; got {el.qualified_name!r}"
    )
    assert el.line_start == 1
    assert el.line_end == 1


def test_dotted_key_path_is_one_element(tmp_path: Path):
    """`a.b.c = 1` is a single key path, not three separate entries."""
    from docgen.catalog_extractor import extract_elements

    f = _write(tmp_path, 'application.conf', 'a.b.c = 1\n')
    elements = extract_elements(f, tmp_path)

    # Should be exactly one element keyed on the full path a.b.c
    assert len(elements) == 1
    assert elements[0].qualified_name.endswith('.a.b.c'), (
        f"dotted-path key should produce one element with the full path; "
        f"got qn={elements[0].qualified_name!r}"
    )


# ---------------------------------------------------------------------------
# Nested block: parent + leaf elements both emitted
# ---------------------------------------------------------------------------


def test_nested_block_emits_parent_and_children(tmp_path: Path):
    """`activation { pub = [...]; licenseFileName = "..." }` should
    produce one element for the `activation` block AND one element
    per leaf inside it."""
    from docgen.catalog_extractor import extract_elements

    src = (
        'activation {\n'
        '  pub = []\n'
        '  licenseFileName = "sblic.af"\n'
        '}\n'
    )
    f = _write(tmp_path, 'reference.conf', src)
    elements = extract_elements(f, tmp_path)

    qns = sorted(el.qualified_name for el in elements)
    # We don't fix the module-qn prefix here; we look at suffixes.
    assert any(qn.endswith('.activation') for qn in qns), (
        f"missing block element 'activation'; got {qns}"
    )
    assert any(qn.endswith('.activation.pub') for qn in qns), (
        f"missing leaf element 'activation.pub'; got {qns}"
    )
    assert any(qn.endswith('.activation.licenseFileName') for qn in qns), (
        f"missing leaf element 'activation.licenseFileName'; got {qns}"
    )


def test_nested_block_parent_qualified_name(tmp_path: Path):
    """Leaves inside a block carry parent_qualified_name pointing at
    the block, not at the file/module."""
    from docgen.catalog_extractor import extract_elements

    src = (
        'activation {\n'
        '  pub = []\n'
        '}\n'
    )
    f = _write(tmp_path, 'reference.conf', src)
    elements = extract_elements(f, tmp_path)

    leaves = [el for el in elements if el.qualified_name.endswith('.activation.pub')]
    assert len(leaves) == 1
    leaf = leaves[0]
    assert leaf.parent_qualified_name is not None
    assert leaf.parent_qualified_name.endswith('.activation'), (
        f"leaf parent_qualified_name should point at 'activation', "
        f"got {leaf.parent_qualified_name!r}"
    )


# ---------------------------------------------------------------------------
# Line numbers
# ---------------------------------------------------------------------------


def test_line_numbers_match_source(tmp_path: Path):
    from docgen.catalog_extractor import extract_elements

    src = (
        '# header comment\n'         # line 1
        'first = 1\n'                # line 2
        '\n'                         # line 3 (blank)
        'second = 2\n'               # line 4
    )
    f = _write(tmp_path, 'reference.conf', src)
    elements = extract_elements(f, tmp_path)

    by_name: dict[str, int] = {
        el.qualified_name.rsplit('.', 1)[-1]: el.line_start
        for el in elements
    }
    assert by_name.get('first') == 2, by_name
    assert by_name.get('second') == 4, by_name


def test_triple_quoted_string_line_range(tmp_path: Path):
    """A triple-quoted multi-line value (the PGP-key form) must have
    line_end > line_start so the catalog correctly reports the value
    spans multiple lines."""
    from docgen.catalog_extractor import extract_elements

    src = (
        'pgp = """-----BEGIN PGP PUBLIC KEY BLOCK-----\n'
        'mQENBFey9YgBEADIMqHXR1aV4qFI...\n'
        '-----END PGP PUBLIC KEY BLOCK-----"""\n'
        'sibling = 1\n'
    )
    f = _write(tmp_path, 'reference.conf', src)
    elements = extract_elements(f, tmp_path)

    pgp_elements = [
        el for el in elements if el.qualified_name.endswith('.pgp')
    ]
    assert len(pgp_elements) == 1
    pgp = pgp_elements[0]
    assert pgp.line_start == 1
    assert pgp.line_end >= 3, (
        f"pgp entry's triple-quoted value spans 3 lines; "
        f"line_end={pgp.line_end}"
    )

    sibling = next(el for el in elements if el.qualified_name.endswith('.sibling'))
    assert sibling.line_start == 4


# ---------------------------------------------------------------------------
# Signature: one-line summary
# ---------------------------------------------------------------------------


def test_signature_is_one_line(tmp_path: Path):
    """`signature` must be a single line — even for entries whose value
    spans multiple lines (triple-quoted strings, blocks). Catalog
    display assumes one-line signatures."""
    from docgen.catalog_extractor import extract_elements

    src = (
        'pgp = """line1\n'
        'line2\n'
        'line3"""\n'
    )
    f = _write(tmp_path, 'reference.conf', src)
    elements = extract_elements(f, tmp_path)

    pgp = next(el for el in elements if el.qualified_name.endswith('.pgp'))
    assert '\n' not in pgp.signature, (
        f"signature must be one line; got {pgp.signature!r}"
    )


# ---------------------------------------------------------------------------
# body_sha: changes when content changes
# ---------------------------------------------------------------------------


def test_body_sha_differs_for_different_content(tmp_path: Path):
    """Two HOCON files whose content differs by even a value change
    must produce different body_sha for the matching key, so the
    catalog can detect modifications across re-extractions."""
    from docgen.catalog_extractor import extract_elements

    a = _write(tmp_path / 'a', 'reference.conf', 'foo = 1\n')
    b = _write(tmp_path / 'b', 'reference.conf', 'foo = 2\n')

    # Use a separate source_root per file so qualified_names align.
    elements_a = extract_elements(a, tmp_path / 'a')
    elements_b = extract_elements(b, tmp_path / 'b')

    sha_a = elements_a[0].body_sha
    sha_b = elements_b[0].body_sha

    assert sha_a and sha_b, 'body_sha must be populated'
    assert sha_a != sha_b, (
        'body_sha should change when value changes; '
        f"got identical sha={sha_a!r}"
    )


# ---------------------------------------------------------------------------
# Real-world fixture: the activation block from scalaproject
# ---------------------------------------------------------------------------


def test_activation_block_full_extraction(tmp_path: Path):
    """End-to-end fixture matching the engine/reference.conf shape.
    All five expected keys must be extracted with correct names and
    line numbers."""
    from docgen.catalog_extractor import extract_elements

    src = (
        'activation {\n'                                       # 1
        '\tpub = [\n'                                          # 2
        '\t\t"""-----BEGIN PGP PUBLIC KEY BLOCK-----\n'        # 3
        '\t\tmQENBGICifMBCACZ...\n'                            # 4
        '\t\t-----END PGP PUBLIC KEY BLOCK-----"""\n'          # 5
        '\t]\n'                                                # 6
        '\tlicenseFileName = "sblic.af"\n'                     # 7
        '\tlicenseSigName = "sblic.af.asc"\n'                  # 8
        '\tlicenseDir = "/var/lib/examplecorp/license"\n'      # 9
        '\tmaxCreationDateDaySpan = 14\n'                      # 10
        '}\n'                                                  # 11
    )
    f = _write(tmp_path, 'reference.conf', src)
    elements = extract_elements(f, tmp_path)

    suffixes = {
        el.qualified_name.rsplit('.activation.', 1)[-1]: el.line_start
        for el in elements
        if '.activation.' in el.qualified_name
    }

    expected = {
        'pub': 2,
        'licenseFileName': 7,
        'licenseSigName': 8,
        'licenseDir': 9,
        'maxCreationDateDaySpan': 10,
    }
    for name, expected_line in expected.items():
        assert name in suffixes, (
            f"missing key 'activation.{name}' in extracted elements: {sorted(suffixes)}"
        )
        assert suffixes[name] == expected_line, (
            f"'activation.{name}' expected on line {expected_line}, "
            f"got line {suffixes[name]}"
        )

    # And the block itself is also emitted.
    block_elements = [
        el for el in elements
        if el.qualified_name.endswith('.activation')
    ]
    assert len(block_elements) == 1, (
        f"expected exactly one 'activation' block element, "
        f"got {len(block_elements)}"
    )
    assert block_elements[0].line_start == 1
    assert block_elements[0].line_end == 11


# ---------------------------------------------------------------------------
# Failure modes: malformed input falls back to []
# ---------------------------------------------------------------------------


def test_malformed_hocon_returns_empty(tmp_path: Path):
    """Parse failures must not crash sync_file_catalog. The extractor
    swallows the error and returns []; the file_index doc still gets
    created, just with zero element_ids."""
    from docgen.catalog_extractor import extract_elements

    # Genuinely broken: starts a block and never closes anything.
    f = _write(tmp_path, 'broken.conf', 'foo { unclosed = 1\nbar = ')
    elements = extract_elements(f, tmp_path)
    assert elements == []


def test_parse_failure_is_logged_not_silent(tmp_path: Path, caplog) -> None:
    """A `.conf` that isn't valid HOCON — a genuinely malformed file, or a
    non-INI dialect (PAM limits.conf, …) — must NOT be silently swallowed.
    The extractor logs at DEBUG naming the file (so a config that degraded
    to file-index-only stays visible when you look) — then returns [] so
    the batch sync keeps going.

    DEBUG, not WARNING: this fallthrough is expected, not an error. INI-style
    `.conf` (a `[section]` header) is routed to the INI extractor upstream,
    so it never reaches this path."""
    import logging

    from docgen.catalog_extractor import extract_elements

    f = _write(tmp_path, 'broken.conf', 'foo { unclosed = 1\nbar = ')
    with caplog.at_level(logging.DEBUG, logger='docgen.hocon_extractor'):
        elements = extract_elements(f, tmp_path)

    assert elements == []
    assert any('broken.conf' in r.getMessage() for r in caplog.records), (
        f"parse failure was not surfaced; logs="
        f"{[r.getMessage() for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# Regression: a real-world reference.conf shape combining substitution-built
# paths (`${base.dir}/log/events`) with the activation.pub PGP-key block.
# Before value-concatenation support, the path lines failed to parse and the
# WHOLE file degraded to file_index-only — so `activation.pub` (the key the
# user expected to find) was never catalogued at all.
# ---------------------------------------------------------------------------


def test_path_concatenation_does_not_suppress_activation_pub(tmp_path: Path):
    from docgen.catalog_extractor import extract_elements

    src = (
        'eventslogger.file.dir {\n'
        '\tparent  = ${base.dir}/log/events\n'
        '\tcurrent = ${eventslogger.file.dir.parent}/current\n'
        '}\n'
        'activation {\n'
        '\tpub = [\n'
        '\t\t"""-----BEGIN PGP PUBLIC KEY BLOCK-----\n'
        '\t\tVersion: GnuPG v2\n'
        '\t\t-----END PGP PUBLIC KEY BLOCK-----"""\n'
        '\t]\n'
        '\tlicenseFileName = "sblic.af"\n'
        '}\n'
    )
    f = _write(tmp_path, 'reference.conf', src)
    elements = extract_elements(f, tmp_path)

    qns = {el.qualified_name for el in elements}
    assert any(qn.endswith('.activation.pub') for qn in qns), (
        f"'activation.pub' must be catalogued even alongside path-concat "
        f"values; got {sorted(qns)}"
    )
    # The path-built keys parse and are catalogued too.
    assert any(qn.endswith('.eventslogger.file.dir.parent') for qn in qns), (
        f"path-concatenation key missing from {sorted(qns)}"
    )
