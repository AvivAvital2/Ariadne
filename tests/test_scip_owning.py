"""The shared owning-symbol resolver (``docgen/scip_owning.py``).

Fixtures mirror REAL scip-python output, not the old synthetic ``scip_symbols``
shape: SymbolInformation.kind is irrelevant (the resolver never reads it), paths
are relative, every definition carries a name-token ``range`` PLUS a separate
body-spanning ``enclosing_range``, and each function emits a *parameter* symbol
whose ``enclosing_range`` spans the whole body. The resolver must attribute a
body-line literal to its enclosing function — never to a parameter (which is not
a persisted graph node) — using 0-indexed lines.
"""
from docgen.scip_extractor import ScipIndex, _ScipDoc, _ScipOccurrence
from docgen.scip_owning import build_owning_resolver

_PFX = 'scip-python python pkg 1.0 '


def _def(symbol: str, name_line: int, body: tuple[int, int] | None = None) -> _ScipOccurrence:
    """A definition occurrence: name-token ``range`` on ``name_line`` (3-tuple),
    plus a multi-line body ``enclosing_range`` (4-tuple) when ``body`` is given."""
    return _ScipOccurrence(
        symbol=_PFX + symbol,
        range=(name_line, 4, 40),
        is_definition=True,
        enclosing_range=(body[0], 0, body[1], 0) if body else (),
    )


def _ref(symbol: str, line: int) -> _ScipOccurrence:
    """A non-definition (reference) occurrence — never an owner."""
    return _ScipOccurrence(symbol=_PFX + symbol, range=(line, 4, 40), is_definition=False)


def test_resolver_attributes_each_line_to_its_real_enclosing_symbol():
    # 0-indexed layout of pkg/mod.py (what scip-python would emit):
    #   0  CONST = "..."          term (name-token only); module spans 0..12
    #   1  def outer(arg):        method, body 1..6 ; `arg` param spans the SAME body
    #   2      x = "..."          -> outer  (param/package/ref/local all excluded)
    #   3      def inner():       method, body 3..5
    #   4          y = "..."      -> inner  (smallest enclosing span)
    #   6      return inner       -> outer
    #   7                         -> None   (module-level; package + malformed skipped)
    #   8  class Cls:             type, body 8..12  -> Cls# (a class IS an owner)
    #   9      attr = "..."       term (name-token only) -> attr, not the class
    #  10      def m(self):       method, body 10..12
    #  11          z = "..."      -> Cls#m  (method wins over the enclosing class)
    occ = (
        _def('`pkg.mod`/', 0, body=(0, 12)),                 # package/module — excluded
        _def('`pkg.mod`/CONST.', 0),                         # term, name-token fallback
        _def('`pkg.mod`/outer().(arg)', 1, body=(1, 6)),     # parameter — must be excluded
        _def('`pkg.mod`/outer().', 1, body=(1, 6)),          # method
        _ref('`pkg.mod`/outer().', 2),                       # reference — non-def, skipped
        _ScipOccurrence(symbol='local 1', range=(2, 4, 5), is_definition=True),  # local
        _def('`pkg.mod`/inner().', 3, body=(3, 5)),          # nested method
        _ScipOccurrence(symbol='weird-no-suffix', range=(7, 0, 7), is_definition=True),  # malformed
        _def('`pkg.mod`/Cls#', 8, body=(8, 12)),             # type (class) — an owner
        _def('`pkg.mod`/Cls#attr.', 9),                      # term (class attribute)
        _def('`pkg.mod`/Cls#m().', 10, body=(10, 12)),       # method inside the class
    )
    index = ScipIndex(documents=(_ScipDoc(relative_path='pkg/mod.py', occurrences=occ),))
    owning = build_owning_resolver(index)

    # module-level term owns its own line (name-token fallback, no body span)
    assert owning('pkg/mod.py', 0) == _PFX + '`pkg.mod`/CONST.'
    # body line of outer -> outer, NOT the parameter (the consistency fix)
    assert owning('pkg/mod.py', 2) == _PFX + '`pkg.mod`/outer().'
    # nested function: smallest enclosing span wins
    assert owning('pkg/mod.py', 4) == _PFX + '`pkg.mod`/inner().'
    # back in outer after inner closes
    assert owning('pkg/mod.py', 6) == _PFX + '`pkg.mod`/outer().'
    # module-level gap -> None (package + malformed symbols never own anything)
    assert owning('pkg/mod.py', 7) is None
    # a class IS an owning container: a class-body line with no smaller owner -> the class
    assert owning('pkg/mod.py', 8) == _PFX + '`pkg.mod`/Cls#'
    # class-level attribute term wins over the enclosing class (smaller span)
    assert owning('pkg/mod.py', 9) == _PFX + '`pkg.mod`/Cls#attr.'
    # a method inside a class wins over the class (smaller span)
    assert owning('pkg/mod.py', 11) == _PFX + '`pkg.mod`/Cls#m().'
    # unknown file -> None
    assert owning('other.py', 1) is None
