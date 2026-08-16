"""Materialize exact source ranges named by compiler-derived citations."""
from __future__ import annotations
import hashlib

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SourceExcerpt:
    source_name: str
    file: str
    line_start: int
    line_end: int
    kind: str
    content: str
    sha256: str


@dataclass(frozen=True)
class SourceMaterialization:
    excerpts: tuple[SourceExcerpt, ...] = field(default_factory=tuple)
    gaps: tuple[str, ...] = field(default_factory=tuple)
def _is_header_line(line: str) -> bool:
    """A comment or annotation line that belongs to the definition below it."""
    stripped = line.strip()
    return stripped.startswith(
        ("//", "/*", "*", "#", "@", '"""', "'''"))
def materialize_citations(
        citations, source_roots, *, expected_hashes=None, extra_ranges=()):
    """Fetch exact definition, call-site, and explicit source ranges.

    Coordinates are one-based and inclusive. The hash covers the complete
    file, binding every excerpt to the snapshot SCIP indexed. A failed range
    is a gap and never partial evidence. Extra ranges accept either a
    four-field single-line tuple or a five-field inclusive-range tuple.
    """
    expected_hashes = expected_hashes or {}
    excerpts: list[SourceExcerpt] = []
    gaps: list[str] = []
    seen: set[tuple[str, str, int, int, str]] = set()
    files: dict[tuple[str, str], tuple[list[str], str] | None] = {}

    def load(source_name: str, file: str):
        key = (source_name, file)
        if key in files:
            return files[key]
        root_value = source_roots.get(source_name)
        if root_value is None:
            gaps.append(f"{source_name}:{file}: source root unavailable")
            files[key] = None
            return None
        root = Path(root_value).resolve()
        raw = Path(file)
        direct = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
        try:
            direct.relative_to(root)
        except ValueError:
            gaps.append(f"{source_name}:{file}: path escapes source root")
            files[key] = None
            return None
        candidates = [direct] if direct.is_file() else []
        if not raw.is_absolute():
            for child in root.iterdir():
                if not child.is_dir():
                    continue
                candidate = (child / raw).resolve()
                try:
                    candidate.relative_to(root)
                except ValueError:
                    continue
                if candidate.is_file() and candidate not in candidates:
                    candidates.append(candidate)
        if len(candidates) > 1:
            gaps.append(f"{source_name}:{file}: ambiguous repository path")
            files[key] = None
            return None
        candidate = candidates[0] if candidates else direct
        try:
            data = candidate.read_bytes()
            text = data.decode("utf-8")
        except (OSError, UnicodeError) as error:
            gaps.append(
                f"{source_name}:{file}: source unavailable "
                f"({type(error).__name__})")
            files[key] = None
            return None
        digest = hashlib.sha256(data).hexdigest()
        expected = expected_hashes.get(file)
        if expected is not None and expected != digest:
            gaps.append(f"{source_name}:{file}: provenance hash mismatch")
            files[key] = None
            return None
        files[key] = (text.splitlines(), digest)
        return files[key]
    def materialize(
            source_name: str, file: str, line_start: int,
            line_end: int, kind: str) -> None:
        key = (source_name, file, line_start, line_end, kind)
        if key in seen:
            return
        seen.add(key)
        loaded = load(source_name, file)
        if loaded is None:
            return
        lines, digest = loaded
        if (
                line_start < 1
                or line_end < line_start
                or line_end > len(lines)):
            gaps.append(
                f"{source_name}:{file}:{line_start}-{line_end}: invalid range")
            return
        excerpts.append(SourceExcerpt(
            source_name=source_name,
            file=file,
            line_start=line_start,
            line_end=line_end,
            kind=kind,
            content="\n".join(lines[line_start - 1:line_end]),
            sha256=digest))
        if kind in ("definition", "definition_body"):
            # Documentation may sit up to two blank lines above the
            # definition it describes; a longer gap means the comment block
            # is not this definition's documentation.
            top = line_start
            cursor = line_start - 1
            blanks = 0
            while cursor >= 1 and line_start - cursor < 60:
                above = lines[cursor - 1]
                if _is_header_line(above):
                    top = cursor
                    blanks = 0
                    cursor -= 1
                    continue
                if not above.strip() and blanks < 2:
                    blanks += 1
                    cursor -= 1
                    continue
                break
            if top < line_start:
                materialize(source_name, file, top, line_start - 1,
                            "doc_header")

    for citation in citations:
        materialize(
            citation.source_name, citation.file,
            citation.line_start, citation.line_start, "definition")
        materialize(
            citation.source_name, citation.call_site_file,
            citation.call_site_line, citation.call_site_line, "call_site")
    for item in extra_ranges:
        if len(item) == 4:
            source_name, file, line, kind = item
            line_start = line_end = int(line)
        elif len(item) == 5:
            source_name, file, line_start, line_end, kind = item
            line_start, line_end = int(line_start), int(line_end)
        else:
            raise ValueError(
                "extra range must contain source, file, line[, line_end], kind")
        materialize(
            str(source_name), str(file), line_start, line_end, str(kind))
    return SourceMaterialization(
        excerpts=tuple(excerpts), gaps=tuple(gaps))
