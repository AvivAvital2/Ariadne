"""Tier 1 of the dry-run explorer: the instant directory/token model.

A pure, recursive scan of a source tree that keeps only the files Ariadne
would *document*, counts their tokens, and rolls the aggregates up a
nested :class:`ScanNode`. No UI, no cost, no writes — instant and fully
testable. Tier 2 attaches per-node cost; Tier 3 renders + navigates it.

"What counts" is defined in exactly one place here:

- **Mapped extensions** = the canonical catalog set
  (:data:`docgen.catalog_writer.CATALOG_EXTS`, what Ariadne documents)
  unioned with every SCIP ``source_extension``
  (:data:`docgen.scip_languages.LANGUAGES`). No hand-rolled list.
- **Pruned directories** are passed in as ``excluded_dirs`` — the caller
  supplies ``Config.resolve_excluded_dirs(source)`` so the ~70 known-junk
  dirs are gone before the user ever sees the tree. Deliberately NOT
  gitignore-based: gitignore answers "what shouldn't be in version
  control," not "what shouldn't be documented."
- **Tokens** come from a per-path counter
  (:func:`docgen.token_count.file_token_counter`, tiktoken). When the
  counter yields ``None`` (tiktoken absent / unreadable file) we fall
  back to the same ``size // CHARS_PER_TOKEN`` heuristic the cost
  estimator uses, so a free offline scan never crashes.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from attrs import frozen

from config import DEFAULTS
from docgen.catalog_writer import CATALOG_EXTS
from docgen.pricing import CHARS_PER_TOKEN
from docgen.scip_languages import LANGUAGES
from docgen.token_count import file_token_counter

# The single definition of "a file Ariadne would document": the canonical
# catalog set ∪ every SCIP source extension. Reusing both canonical lists
# means a new language or doc type added there flows here for free.
MAPPED_EXTENSIONS: frozenset[str] = frozenset(CATALOG_EXTS) | frozenset(
    ext for lang in LANGUAGES for ext in lang.source_extensions
)


def is_mapped(path: "Path | str") -> bool:
    """True when ``path``'s suffix is one Ariadne documents."""
    suffix = path.suffix if isinstance(path, Path) else Path(path).suffix
    return suffix.lower() in MAPPED_EXTENSIONS


@frozen
class ScanNode:
    """One node in the scanned tree. Aggregates roll up from children.

    ``mapped_files`` / ``content_tokens`` count only documented files;
    ``total_bytes`` counts *every* file on disk so a large data dir with
    few mapped files still stands out as an on-disk hog.
    """

    name: str
    rel_path: str
    is_dir: bool
    mapped_files: int
    content_tokens: int
    total_bytes: int
    children: tuple["ScanNode", ...] = ()


def scan_tree(
    root: "Path | str",
    *,
    excluded_dirs,
    token_counter: "Callable[[Path], int | None] | None" = None,
) -> ScanNode:
    """Recursively scan ``root`` into a nested :class:`ScanNode`.

    ``excluded_dirs`` is a set of directory *names* pruned at every depth
    (pass ``Config.resolve_excluded_dirs(source)``). ``token_counter``
    defaults to a fresh tiktoken counter for the configured model; inject
    one in tests. Symlinks are never followed and unreadable entries are
    skipped, so a scan never crashes a free preview.
    """
    if token_counter is None:
        token_counter = file_token_counter(DEFAULTS['model'])
    root_path = Path(root)
    return _scan_dir(root_path, root_path, excluded_dirs, token_counter)


def _scan_dir(path, root, excluded_dirs, token_counter) -> ScanNode:
    """Build the node for directory ``path`` and recurse into it."""
    children: list[ScanNode] = []
    try:
        entries = list(path.iterdir())
    except OSError:
        entries = []
    for entry in entries:
        if entry.is_symlink():
            continue  # never follow symlinks
        if entry.is_dir():
            if entry.name in excluded_dirs:
                continue  # prune policy dir + its whole subtree
            children.append(_scan_dir(entry, root, excluded_dirs, token_counter))
        elif entry.is_file():
            children.append(_file_node(entry, root, token_counter))
    # Directories first, then files; each group by content_tokens desc.
    children.sort(key=lambda n: (not n.is_dir, -n.content_tokens))
    return ScanNode(
        name=path.name,
        rel_path=_rel(path, root),
        is_dir=True,
        mapped_files=sum(c.mapped_files for c in children),
        content_tokens=sum(c.content_tokens for c in children),
        total_bytes=sum(c.total_bytes for c in children),
        children=tuple(children),
    )


def _file_node(path, root, token_counter) -> ScanNode:
    """Build a leaf node for a single file.

    Reached only after ``is_file()`` confirmed a successful stat, so the
    stat here cannot fail short of a TOCTOU race; unreadable files are
    already filtered upstream (``is_file()`` returns ``False`` for them).
    """
    size = path.stat().st_size
    if is_mapped(path):
        return ScanNode(
            name=path.name,
            rel_path=_rel(path, root),
            is_dir=False,
            mapped_files=1,
            content_tokens=_count_tokens(path, size, token_counter),
            total_bytes=size,
        )
    # Non-mapped: still counted in total_bytes so a big data dir shows up.
    return ScanNode(
        name=path.name,
        rel_path=_rel(path, root),
        is_dir=False,
        mapped_files=0,
        content_tokens=0,
        total_bytes=size,
    )


def _count_tokens(path, size, token_counter) -> int:
    """Tokens for ``path`` via the counter, falling back to the same
    ``size // CHARS_PER_TOKEN`` heuristic the cost estimator uses when the
    counter yields ``None`` (tiktoken absent) or raises (unreadable)."""
    try:
        tokens = token_counter(path)
    except Exception:  # noqa: BLE001 — any counter failure → char fallback
        tokens = None
    if tokens is None:
        return int(size // CHARS_PER_TOKEN)
    return tokens


def _rel(path, root) -> str:
    """``path`` relative to the scan root (``'.'`` for the root itself)."""
    return str(path.relative_to(root))
