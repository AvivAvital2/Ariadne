"""Warning-suppressed ``ast.parse`` for parsing TARGET source code.

Indexers parse arbitrary third-party Python. Such source routinely contains
string literals with invalid escape sequences (``"\\c"``, ``r``-less regexes,
Windows-ish paths) that Python 3.12+ reports as ``SyntaxWarning``. Those are
warnings about the *parsed* code, not about Ariadne — noise the user can't act
on and shouldn't see flooding ``ariadne index`` / ``sync`` / ``onboard``.

Parse target source through :func:`safe_ast_parse` so that noise never reaches
the console, instead of remembering to install a process-wide ``filterwarnings``
at each command entry point (which is exactly how the ``index`` path got missed).

This lives at the repo root (a stdlib-only leaf) so both ``docgen`` and
``library`` can import it without triggering either package's ``__init__`` — a
``library`` -> ``docgen.*`` import would otherwise cycle back through
``docgen.orchestrator`` -> ``library``.
"""
from __future__ import annotations

import ast
import warnings


def safe_ast_parse(source: str, filename: str = '<unknown>') -> ast.Module:
    """``ast.parse(source, filename)`` with ``SyntaxWarning`` suppressed.

    Only lexical warnings about the *parsed* source (invalid escapes and the
    like) are silenced; a genuine ``SyntaxError`` still propagates. ``filename``
    is forwarded so any error that does raise carries an accurate location.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)
        return ast.parse(source, filename=filename)
