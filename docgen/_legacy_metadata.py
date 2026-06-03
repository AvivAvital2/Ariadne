"""Legacy data structures for Python source code metadata (deprecated).

Kept for emergency rollback through release N+1; will be deleted once
``OrchestratorConfig.catalog_only_generator`` has been the default for at
least one release. New code should use ``docgen.catalog_enrich`` types
(``EnrichedFileBundle``, ``PythonEnrichment``, ``StructuredImport``)
instead, which work for every supported language, not just Python.

This file was renamed from ``docgen/metadata.py`` in Catalog transition
Phase 4.1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from attrs import frozen


@frozen
class ArgumentInfo:
    """Information about a function or method argument."""

    name: str
    annotation: str | None = None
    default: str | None = None
    kind: Literal['positional', 'keyword', 'var_positional', 'var_keyword'] = 'positional'


@frozen
class FunctionInfo:
    """Information about a function or method definition."""

    name: str
    lineno: int
    docstring: str | None = None
    arguments: tuple[ArgumentInfo, ...] = ()
    return_annotation: str | None = None
    decorators: tuple[str, ...] = ()
    is_async: bool = False
    is_method: bool = False
    is_classmethod: bool = False
    is_staticmethod: bool = False
    is_property: bool = False

    @property
    def signature(self) -> str:
        """Generate a readable function signature."""
        args = []
        for arg in self.arguments:
            part = arg.name
            if arg.annotation:
                part = f'{part}: {arg.annotation}'
            if arg.default:
                part = f'{part} = {arg.default}'
            if arg.kind == 'var_positional':
                part = f'*{part}'
            elif arg.kind == 'var_keyword':
                part = f'**{part}'
            args.append(part)

        sig = f"{'async ' if self.is_async else ''}def {self.name}({', '.join(args)})"
        if self.return_annotation:
            sig = f'{sig} -> {self.return_annotation}'
        return sig


@frozen
class ClassInfo:
    """Information about a class definition."""

    name: str
    lineno: int
    docstring: str | None = None
    bases: tuple[str, ...] = ()
    decorators: tuple[str, ...] = ()
    methods: tuple[FunctionInfo, ...] = ()
    class_variables: tuple[tuple[str, str | None], ...] = ()  # (name, annotation)
    is_dataclass: bool = False
    is_attrs: bool = False
    is_abstract: bool = False

    @property
    def public_methods(self) -> tuple[FunctionInfo, ...]:
        """Get public methods (not starting with underscore)."""
        return tuple(m for m in self.methods if not m.name.startswith('_'))

    @property
    def special_methods(self) -> tuple[FunctionInfo, ...]:
        """Get special dunder methods."""
        return tuple(m for m in self.methods if m.name.startswith('__') and m.name.endswith('__'))


@frozen
class ImportInfo:
    """Information about an import statement."""

    module: str
    names: tuple[str, ...] = ()  # For "from X import a, b" - the names imported
    alias: str | None = None  # For "import X as Y" or "from X import a as b"
    is_from_import: bool = False
    lineno: int = 0

    @property
    def imported_names(self) -> tuple[str, ...]:
        """Get the names that are made available by this import."""
        if self.is_from_import:
            return self.names
        if self.alias:
            return (self.alias,)
        # For "import foo.bar", only "foo" is available
        return (self.module.split('.')[0],)


@frozen
class ModuleMetadata:
    """Comprehensive metadata about a Python module."""

    path: Path
    module_name: str
    docstring: str | None = None
    imports: tuple[ImportInfo, ...] = ()
    classes: tuple[ClassInfo, ...] = ()
    functions: tuple[FunctionInfo, ...] = ()
    module_variables: tuple[tuple[str, str | None], ...] = ()  # (name, annotation)
    source_hash: str = ''  # SHA256 hash for staleness detection
    line_count: int = 0

    @property
    def public_classes(self) -> tuple[ClassInfo, ...]:
        """Get public classes (not starting with underscore)."""
        return tuple(c for c in self.classes if not c.name.startswith('_'))

    @property
    def public_functions(self) -> tuple[FunctionInfo, ...]:
        """Get public functions (not starting with underscore)."""
        return tuple(f for f in self.functions if not f.name.startswith('_'))

    @property
    def all_public_names(self) -> tuple[str, ...]:
        """Get all public names defined in this module."""
        names: list[str] = []
        names.extend(c.name for c in self.public_classes)
        names.extend(f.name for f in self.public_functions)
        names.extend(name for name, _ in self.module_variables if not name.startswith('_'))
        return tuple(sorted(set(names)))

    @property
    def dependencies(self) -> tuple[str, ...]:
        """Get unique module dependencies from imports."""
        deps = set()
        for imp in self.imports:
            # Get top-level module
            top_module = imp.module.split('.')[0]
            deps.add(top_module)
        return tuple(sorted(deps))

    def summary(self, max_length: int = 500) -> str:
        """Generate a brief summary of the module."""
        parts = [f'Module: {self.module_name}']
        if self.docstring:
            # First line of docstring
            first_line = self.docstring.split('\n')[0].strip()
            parts.append(f'Description: {first_line}')

        if self.public_classes:
            class_names = ', '.join(c.name for c in self.public_classes[:5])
            if len(self.public_classes) > 5:
                class_names += f' (+{len(self.public_classes) - 5} more)'
            parts.append(f'Classes: {class_names}')

        if self.public_functions:
            func_names = ', '.join(f.name for f in self.public_functions[:5])
            if len(self.public_functions) > 5:
                func_names += f' (+{len(self.public_functions) - 5} more)'
            parts.append(f'Functions: {func_names}')

        summary = '\n'.join(parts)
        if len(summary) > max_length:
            summary = summary[: max_length - 3] + '...'
        return summary


@frozen
class ModuleGroup:
    """A group of related modules (e.g., a package or subpackage)."""

    name: str
    path: Path
    modules: tuple[ModuleMetadata, ...] = ()
    subgroups: tuple['ModuleGroup', ...] = ()
    init_module: ModuleMetadata | None = None

    @property
    def all_modules(self) -> tuple[ModuleMetadata, ...]:
        """Get all modules including those in subgroups."""
        all_mods = list(self.modules)
        if self.init_module:
            all_mods.append(self.init_module)
        for subgroup in self.subgroups:
            all_mods.extend(subgroup.all_modules)
        return tuple(all_mods)

    @property
    def all_public_classes(self) -> tuple[ClassInfo, ...]:
        """Get all public classes across all modules."""
        classes = []
        for mod in self.all_modules:
            classes.extend(mod.public_classes)
        return tuple(classes)

    @property
    def all_public_functions(self) -> tuple[FunctionInfo, ...]:
        """Get all public functions across all modules."""
        funcs = []
        for mod in self.all_modules:
            funcs.extend(mod.public_functions)
        return tuple(funcs)

    def summary(self) -> str:
        """Generate a brief summary of the module group."""
        parts = [f'Package: {self.name}']
        parts.append(f'Modules: {len(self.all_modules)}')
        parts.append(f'Classes: {len(self.all_public_classes)}')
        parts.append(f'Functions: {len(self.all_public_functions)}')
        return '\n'.join(parts)
