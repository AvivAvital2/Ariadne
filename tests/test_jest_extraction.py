"""Jest / BDD test-structure extraction: describe/it/test → catalog elements.

In-house ast-grep recognition in ``docgen.catalog_extractor`` (see
``designs/jest-test-extraction.md``). Synthetic fixtures only.
"""
from __future__ import annotations

from docgen.catalog_extractor import extract_elements


def _elements_for(tmp_path, filename: str, src: str):
    spec = tmp_path / filename
    spec.write_text(src, encoding='utf-8')
    return extract_elements(spec, tmp_path)


def test_bdd_blocks_extracted(tmp_path):
    """describe/it/test become located, correctly-nested catalog elements:
    describe → js_test_suite, it/test → js_test_case; nesting flows through
    parent_qualified_name; non-test calls and non-string-named blocks are
    skipped; a test under a dynamically-named suite attaches to the nearest
    statically-named scope (here, the module)."""
    src = (
        "describe('Widget', () => {\n"            # 1: top-level suite
        "  it('renders', () => { helper(); });\n" # 2: case (+ a non-test call)
        "  test('updates', () => {});\n"          # 3: case via test() alias
        "  it();\n"                               # 4: no name -> skipped
        "  describe('events', () => {\n"          # 5: nested suite
        "    it('emits change', () => {});\n"     # 6: nested case
        "  });\n"
        "});\n"
        "describe(dynamicName, () => {\n"         # 9: dynamic name -> suite skipped
        "  it('orphan', () => {});\n"             # 10: case re-homed to module
        "});\n"
    )
    elements = _elements_for(tmp_path, 'widget.spec.js', src)
    suites = {e.qualified_name: e for e in elements if e.subtype == 'js_test_suite'}
    cases = {e.qualified_name: e for e in elements if e.subtype == 'js_test_case'}

    # Suites: top-level 'Widget' and nested 'events'; the dynamically-named
    # describe is not emitted.
    assert set(suites) == {'widget.spec::Widget', 'widget.spec::Widget > events'}
    assert suites['widget.spec::Widget'].language == 'javascript'
    assert suites['widget.spec::Widget'].line_start == 1
    events = suites['widget.spec::Widget > events']
    assert events.parent_qualified_name == 'widget.spec::Widget'
    assert events.line_start == 5

    # Cases: it()/test() at top level, the nested it(), and the orphan under the
    # dynamic suite (re-homed to the module). The nameless it() is skipped.
    assert set(cases) == {
        'widget.spec::Widget > renders',
        'widget.spec::Widget > updates',
        'widget.spec::Widget > events > emits change',
        'widget.spec::orphan',
    }
    assert cases['widget.spec::Widget > renders'].parent_qualified_name == 'widget.spec::Widget'
    assert cases['widget.spec::Widget > renders'].line_start == 2
    nested = cases['widget.spec::Widget > events > emits change']
    assert nested.parent_qualified_name == 'widget.spec::Widget > events'
    assert nested.line_start == 6
    # dynamically-named suite skipped, but its case re-homes to the module scope
    assert cases['widget.spec::orphan'].parent_qualified_name == 'widget.spec'
    assert cases['widget.spec::orphan'].line_start == 10
