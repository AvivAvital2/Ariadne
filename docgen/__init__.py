"""Documentation generation module for Claude Library.

This module provides tools for automatically generating documentation
from source code using LLM analysis.

The legacy ``analyzer`` and ``metadata`` modules were renamed to
``_legacy_analyzer`` and ``_legacy_metadata`` in Catalog transition Phase 4.
They are no longer part of the public API; new code should use
``docgen.catalog_extractor`` + ``docgen.catalog_enrich`` instead, which
are language-agnostic.
"""

from docgen.catalog_enrich import (
    EnrichedElementInfo,
    EnrichedFileBundle,
    PythonEnrichment,
    StructuredImport,
    enrich_file,
    enrich_python_elements,
)
from docgen.catalog_extractor import ElementInfo, extract_elements
from docgen.crossref import CrossRefDetector
from docgen.generator import DocGenerator, GeneratorConfig
from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig
from docgen.staleness import SourceRecord, StalenessTracker
from docgen.validator import ContentValidator, ValidationResult

__all__ = [
    # Catalog (multi-language structural index)
    'ElementInfo',
    'extract_elements',
    'EnrichedElementInfo',
    'EnrichedFileBundle',
    'PythonEnrichment',
    'StructuredImport',
    'enrich_file',
    'enrich_python_elements',
    # Generator
    'DocGenerator',
    'GeneratorConfig',
    # Staleness
    'StalenessTracker',
    'SourceRecord',
    # Cross-references
    'CrossRefDetector',
    # Validator
    'ContentValidator',
    'ValidationResult',
    # Orchestrator
    'DocGenOrchestrator',
    'OrchestratorConfig',
]
