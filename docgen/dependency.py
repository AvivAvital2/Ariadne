"""Dependency detection for documentation sources.

This module provides tools for detecting dependencies between documentation
sources by scanning Python files for import statements that reference
other known sources.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

from attrs import frozen


@frozen
class ImportEvidence:
    """Evidence of a dependency from an import statement.

    Attributes:
        file_path: Relative path to file containing the import.
        line_number: Line number of the import.
        import_statement: The actual import line.
        source_name: Which source this imports from.
    """

    file_path: str
    line_number: int
    import_statement: str
    source_name: str


@frozen
class DetectedDependency:
    """A detected dependency with supporting evidence.

    Attributes:
        source_name: The dependency source name.
        evidence: List of ImportEvidence examples (up to max_evidence).
    """

    source_name: str
    evidence: tuple[ImportEvidence, ...]


def _extract_imports_with_context(
    py_file: Path,
    known_sources: dict[str, Path],
) -> list[tuple[int, str, str]]:
    """Extract import statements from a Python file with line numbers.

    Args:
        py_file: Path to the Python file.
        known_sources: Dict mapping source names to their paths.

    Returns:
        List of tuples: (line_number, import_statement, matching_source_name)
    """
    try:
        content = py_file.read_text()
        lines = content.split('\n')
    except (OSError, UnicodeDecodeError):
        return []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    results = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                source = _match_import_to_source(module_name, known_sources)
                if source:
                    line_num = node.lineno
                    import_stmt = lines[line_num - 1].strip() if line_num <= len(lines) else ''
                    results.append((line_num, import_stmt, source))

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module
                source = _match_import_to_source(module_name, known_sources)
                if source:
                    line_num = node.lineno
                    import_stmt = lines[line_num - 1].strip() if line_num <= len(lines) else ''
                    results.append((line_num, import_stmt, source))

    return results


def _match_import_to_source(
    module_name: str,
    known_sources: dict[str, Path],
) -> str | None:
    """Match an import module name to a known source.

    Args:
        module_name: The module name from the import statement.
        known_sources: Dict mapping source names to their paths.

    Returns:
        The source name if matched, None otherwise.
    """
    # Get the top-level package name
    top_level = module_name.split('.', maxsplit=1)[0]

    # Check if it matches any known source name directly
    if top_level in known_sources:
        return top_level

    # Check if any source directory contains a package with this name
    for source_name, source_path in known_sources.items():
        # Check if the source path contains a matching package
        if source_path.exists():
            # Check if source_path itself is the package
            if source_path.name == top_level:
                return source_name
            # Check for subdirectory matching the top-level name
            potential_pkg = source_path / top_level
            if potential_pkg.exists() and (potential_pkg / '__init__.py').exists():
                return source_name
            # Check parent directory (source might be inside the package)
            if (source_path / '__init__.py').exists():
                # source_path is itself a package - check if its name matches
                if source_path.name == top_level:
                    return source_name

    return None


def detect_dependencies(
    source_path: Path,
    known_sources: dict[str, Path],
    max_evidence: int = 3,
    ignore: frozenset[str] = frozenset(),
) -> list[DetectedDependency]:
    """Scan Python files for imports matching known sources.

    This function walks through all Python files in the source directory
    and identifies imports that reference other known documentation sources.

    Args:
        source_path: Path to the source code directory to scan.
        known_sources: Dict mapping source names to their resolved paths.
        max_evidence: Maximum number of evidence examples per dependency.

    Returns:
        List of DetectedDependency objects with evidence.
    """
    evidence_by_source: dict[str, list[ImportEvidence]] = defaultdict(list)

    # Filter out the source being scanned from known_sources
    other_sources = {
        name: path
        for name, path in known_sources.items()
        if path.resolve() != source_path.resolve()
    }
    other_sources = {n: p for n, p in other_sources.items() if n not in ignore}

    if not other_sources:
        return []

    # Scan all Python files
    for py_file in source_path.rglob('*.py'):
        # Skip test files and __pycache__
        rel_path = py_file.relative_to(source_path)
        if '__pycache__' in rel_path.parts:
            continue
        if 'test' in py_file.stem.lower() and py_file.stem.startswith('test'):
            continue

        imports = _extract_imports_with_context(py_file, other_sources)

        for line_num, import_stmt, source_name in imports:
            evidence_by_source[source_name].append(
                ImportEvidence(
                    file_path=str(rel_path),
                    line_number=line_num,
                    import_statement=import_stmt,
                    source_name=source_name,
                )
            )

    # Build results with limited evidence
    return [
        DetectedDependency(
            source_name=name,
            evidence=tuple(evidence[:max_evidence]),
        )
        for name, evidence in sorted(evidence_by_source.items())
    ]
