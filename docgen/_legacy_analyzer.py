"""Legacy AST-based Python source code analyzer (deprecated).

Kept for emergency rollback through release N+1; will be deleted once
``OrchestratorConfig.catalog_only_generator`` has been the default for at
least one release. New code should not import from this module — use
``docgen.catalog_extractor`` + ``docgen.catalog_enrich`` instead, which
support every supported language, not just Python.

This file was renamed from ``docgen/analyzer.py`` in Catalog transition
Phase 4.1.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from attrs import define, field

from docgen._legacy_metadata import (
    ArgumentInfo,
    ClassInfo,
    FunctionInfo,
    ImportInfo,
    ModuleGroup,
    ModuleMetadata,
)


def _compute_hash(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def _get_annotation_str(node: ast.expr | None) -> str | None:
    """Convert an AST annotation node to a string representation."""
    if node is None:
        return None
    return ast.unparse(node)


def _get_decorator_names(decorator_list: list[ast.expr]) -> tuple[str, ...]:
    """Extract decorator names from decorator list."""
    names = []
    for dec in decorator_list:
        if isinstance(dec, ast.Name):
            names.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            names.append(ast.unparse(dec))
        elif isinstance(dec, ast.Call):
            if isinstance(dec.func, ast.Name):
                names.append(dec.func.id)
            elif isinstance(dec.func, ast.Attribute):
                names.append(ast.unparse(dec.func))
            else:
                names.append(ast.unparse(dec))
        else:
            names.append(ast.unparse(dec))
    return tuple(names)


def _is_dataclass(decorators: tuple[str, ...]) -> bool:
    """Check if decorators indicate a dataclass."""
    return any(d in ('dataclass', 'dataclasses.dataclass') for d in decorators)


def _is_attrs_class(decorators: tuple[str, ...]) -> bool:
    """Check if decorators indicate an attrs class."""
    attrs_decorators = {'define', 'frozen', 'attrs', 'attr.s', 'attrs.define', 'attrs.frozen'}
    return any(d in attrs_decorators for d in decorators)


def _is_abstract_class(bases: tuple[str, ...], decorators: tuple[str, ...]) -> bool:
    """Check if class is abstract (inherits ABC or uses abstractmethod)."""
    if 'ABC' in bases or 'abc.ABC' in bases:
        return True
    return 'abstractmethod' in decorators or 'abc.abstractmethod' in decorators


@define
class SourceAnalyzer:
    """Analyzes Python source files to extract structured metadata.

    This class uses AST parsing to extract information about modules,
    classes, functions, imports, and other code structures.

    Attributes:
        exclude_patterns: Glob patterns for files to exclude.
        include_private: Whether to include private members (starting with _).
    """

    exclude_patterns: tuple[str, ...] = field(default=('**/test_*.py', '**/*_test.py', '**/conftest.py'))
    include_private: bool = True

    def analyze_file(self, path: Path) -> ModuleMetadata:
        """Analyze a single Python file and extract metadata.

        Args:
            path: Path to the Python file.

        Returns:
            ModuleMetadata containing extracted information.

        Raises:
            SyntaxError: If the file contains invalid Python syntax.
            FileNotFoundError: If the file does not exist.
        """
        content = path.read_text(encoding='utf-8')
        source_hash = _compute_hash(content)
        line_count = len(content.splitlines())

        tree = ast.parse(content, filename=str(path))

        module_name = self._path_to_module_name(path)
        docstring = ast.get_docstring(tree)

        imports = self._extract_imports(tree)
        classes = self._extract_classes(tree)
        functions = self._extract_functions(tree)
        module_vars = self._extract_module_variables(tree)

        return ModuleMetadata(
            path=path,
            module_name=module_name,
            docstring=docstring,
            imports=imports,
            classes=classes,
            functions=functions,
            module_variables=module_vars,
            source_hash=source_hash,
            line_count=line_count,
        )

    def analyze_directory(self, path: Path, recursive: bool = True) -> ModuleGroup:
        """Analyze a directory of Python files.

        Args:
            path: Path to the directory.
            recursive: Whether to recursively analyze subdirectories.

        Returns:
            ModuleGroup containing all analyzed modules.
        """
        if not path.is_dir():
            msg = f'Path is not a directory: {path}'
            raise ValueError(msg)

        modules: list[ModuleMetadata] = []
        subgroups: list[ModuleGroup] = []
        init_module: ModuleMetadata | None = None

        # Check if this is a Python package
        init_path = path / '__init__.py'
        if init_path.exists():
            try:
                init_module = self.analyze_file(init_path)
            except SyntaxError:
                pass  # Skip files with syntax errors

        # Process Python files
        for py_file in sorted(path.glob('*.py')):
            if py_file.name == '__init__.py':
                continue  # Already processed
            if self._should_exclude(py_file):
                continue
            try:
                modules.append(self.analyze_file(py_file))
            except SyntaxError:
                pass  # Skip files with syntax errors

        # Process subdirectories
        if recursive:
            for subdir in sorted(path.iterdir()):
                if not subdir.is_dir():
                    continue
                if subdir.name.startswith(('.', '_')) and subdir.name != '__pycache__':
                    # Skip hidden dirs but check for __init__.py
                    pass
                if subdir.name == '__pycache__':
                    continue
                if (subdir / '__init__.py').exists():
                    subgroups.append(self.analyze_directory(subdir, recursive=True))

        return ModuleGroup(
            name=path.name,
            path=path,
            modules=tuple(modules),
            subgroups=tuple(subgroups),
            init_module=init_module,
        )

    def get_dependencies(self, metadata: ModuleMetadata) -> set[str]:
        """Extract module dependencies from imports.

        Args:
            metadata: Module metadata to analyze.

        Returns:
            Set of top-level module names that this module depends on.
        """
        deps = set()
        for imp in metadata.imports:
            # Extract top-level module
            top_module = imp.module.split('.')[0]
            deps.add(top_module)
        return deps

    def get_internal_dependencies(self, metadata: ModuleMetadata, package_name: str) -> set[str]:
        """Extract internal package dependencies from imports.

        Args:
            metadata: Module metadata to analyze.
            package_name: Name of the package to filter for.

        Returns:
            Set of internal module names that this module depends on.
        """
        deps = set()
        for imp in metadata.imports:
            if imp.module.startswith(package_name):
                deps.add(imp.module)
        return deps

    def _path_to_module_name(self, path: Path) -> str:
        """Convert a file path to a Python module name."""
        # Try to find a parent with __init__.py to determine package structure
        parts = []
        current = path
        if current.name.endswith('.py'):
            name = current.stem
            if name != '__init__':
                parts.append(name)
            current = current.parent

        while current != current.parent:
            if (current / '__init__.py').exists():
                parts.append(current.name)
                current = current.parent
            else:
                break

        return '.'.join(reversed(parts)) if parts else path.stem

    def _should_exclude(self, path: Path) -> bool:
        """Check if a file should be excluded based on patterns."""
        for pattern in self.exclude_patterns:
            if path.match(pattern):
                return True
        return False

    def _extract_imports(self, tree: ast.Module) -> tuple[ImportInfo, ...]:
        """Extract import statements from AST."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(
                        ImportInfo(
                            module=alias.name,
                            alias=alias.asname,
                            is_from_import=False,
                            lineno=node.lineno,
                        )
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                names = tuple(alias.name for alias in node.names)
                imports.append(
                    ImportInfo(
                        module=module,
                        names=names,
                        is_from_import=True,
                        lineno=node.lineno,
                    )
                )
        return tuple(imports)

    def _extract_classes(self, tree: ast.Module) -> tuple[ClassInfo, ...]:
        """Extract class definitions from AST."""
        classes = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                if not self.include_private and node.name.startswith('_'):
                    continue
                classes.append(self._parse_class(node))
        return tuple(classes)

    def _parse_class(self, node: ast.ClassDef) -> ClassInfo:
        """Parse a class definition node."""
        docstring = ast.get_docstring(node)
        bases = tuple(ast.unparse(base) for base in node.bases)
        decorators = _get_decorator_names(node.decorator_list)

        methods = []
        class_vars = []

        for item in node.body:
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                if not self.include_private and item.name.startswith('_') and not item.name.startswith('__'):
                    continue
                methods.append(self._parse_function(item, is_method=True, class_decorators=decorators))
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                annotation = _get_annotation_str(item.annotation)
                class_vars.append((item.target.id, annotation))

        return ClassInfo(
            name=node.name,
            lineno=node.lineno,
            docstring=docstring,
            bases=bases,
            decorators=decorators,
            methods=tuple(methods),
            class_variables=tuple(class_vars),
            is_dataclass=_is_dataclass(decorators),
            is_attrs=_is_attrs_class(decorators),
            is_abstract=_is_abstract_class(bases, decorators),
        )

    def _extract_functions(self, tree: ast.Module) -> tuple[FunctionInfo, ...]:
        """Extract module-level function definitions from AST."""
        functions = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                if not self.include_private and node.name.startswith('_'):
                    continue
                functions.append(self._parse_function(node, is_method=False))
        return tuple(functions)

    def _parse_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        is_method: bool = False,
        class_decorators: tuple[str, ...] = (),
    ) -> FunctionInfo:
        """Parse a function definition node."""
        docstring = ast.get_docstring(node)
        decorators = _get_decorator_names(node.decorator_list)
        return_annotation = _get_annotation_str(node.returns)
        is_async = isinstance(node, ast.AsyncFunctionDef)

        # Parse arguments
        arguments = self._parse_arguments(node.args)

        # Check for method types
        is_classmethod = 'classmethod' in decorators
        is_staticmethod = 'staticmethod' in decorators
        is_property = 'property' in decorators or any(d.endswith('.getter') for d in decorators)

        return FunctionInfo(
            name=node.name,
            lineno=node.lineno,
            docstring=docstring,
            arguments=arguments,
            return_annotation=return_annotation,
            decorators=decorators,
            is_async=is_async,
            is_method=is_method,
            is_classmethod=is_classmethod,
            is_staticmethod=is_staticmethod,
            is_property=is_property,
        )

    def _parse_arguments(self, args: ast.arguments) -> tuple[ArgumentInfo, ...]:
        """Parse function arguments."""
        arguments: list[ArgumentInfo] = []

        # Calculate defaults offset
        num_positional = len(args.posonlyargs) + len(args.args)
        num_defaults = len(args.defaults)
        default_offset = num_positional - num_defaults

        # Process positional-only args
        for i, arg in enumerate(args.posonlyargs):
            default_idx = i - default_offset
            default = None
            if default_idx >= 0 and default_idx < len(args.defaults):
                default = ast.unparse(args.defaults[default_idx])
            arguments.append(
                ArgumentInfo(
                    name=arg.arg,
                    annotation=_get_annotation_str(arg.annotation),
                    default=default,
                    kind='positional',
                )
            )

        # Process regular positional args
        for i, arg in enumerate(args.args):
            default_idx = i + len(args.posonlyargs) - default_offset
            default = None
            if default_idx >= 0 and default_idx < len(args.defaults):
                default = ast.unparse(args.defaults[default_idx])
            arguments.append(
                ArgumentInfo(
                    name=arg.arg,
                    annotation=_get_annotation_str(arg.annotation),
                    default=default,
                    kind='positional',
                )
            )

        # Process *args
        if args.vararg:
            arguments.append(
                ArgumentInfo(
                    name=args.vararg.arg,
                    annotation=_get_annotation_str(args.vararg.annotation),
                    kind='var_positional',
                )
            )

        # Process keyword-only args
        for i, arg in enumerate(args.kwonlyargs):
            default = None
            if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
                default = ast.unparse(args.kw_defaults[i])
            arguments.append(
                ArgumentInfo(
                    name=arg.arg,
                    annotation=_get_annotation_str(arg.annotation),
                    default=default,
                    kind='keyword',
                )
            )

        # Process **kwargs
        if args.kwarg:
            arguments.append(
                ArgumentInfo(
                    name=args.kwarg.arg,
                    annotation=_get_annotation_str(args.kwarg.annotation),
                    kind='var_keyword',
                )
            )

        return tuple(arguments)

    def _extract_module_variables(self, tree: ast.Module) -> tuple[tuple[str, str | None], ...]:
        """Extract module-level variable assignments."""
        variables: list[tuple[str, str | None]] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name = node.target.id
                if not self.include_private and name.startswith('_'):
                    continue
                annotation = _get_annotation_str(node.annotation)
                variables.append((name, annotation))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        if not self.include_private and name.startswith('_'):
                            continue
                        # Try to infer type from value
                        variables.append((name, None))
        return tuple(variables)
