"""Tests for docgen.catalog_lookup."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from docgen import catalog_lookup
from docgen.catalog_lookup import fuzzy_suggestions, list_elements_in_file, lookup_symbol
from docgen.catalog_writer import _element_doc_id
from library import Library


@pytest.fixture                                                                                                                                                                      
def library(tmp_path: Path):
    lib = Library(tmp_path / 'lookup-test.db')
    yield lib                                                                                                                                                                        
    lib.close()
                                                                                                                                                                                     
                
def _add_element(
    library,
    source_name: str,                                                                                                                                                                
    qualified_name: str,
    *,                                                                                                                                                                               
    file: str = '/src/x.py',
    language: str = 'python',
    subtype: str = 'function',                                                                                                                                                       
    description: str | None = None,
):                                                                                                                                                                                   
    meta = {    
        'kind': 'element',
        'source_name': source_name,
        'language': language,                                                                                                                                                        
        'subtype': subtype,
        'qualified_name': qualified_name,                                                                                                                                            
        'signature': f"def {qualified_name.rsplit('.', maxsplit=1)[-1]}()",
        'location': {'line_start': 1, 'line_end': 2, 'col_start': 0, 'col_end': 0},                                                                                                  
        'parent_qualified_name': None,                                                                                                                                               
        'sha_at_sync': 'abc',                                                                                                                                                        
    }                                                                                                                                                                                
    if description is not None:
        meta['description'] = description                                                                                                                                            
    return library.add_document(
        content_type='catalog',                                                                                                                                                      
        title=qualified_name,
        content=f'function {qualified_name}',                                                                                                                                        
        source_files=[file],
        embedding=np.zeros(1536, dtype=np.float32),                                                                                                                                  
        metadata=meta,
        doc_id=_element_doc_id(source_name, qualified_name),                                                                                                                         
    )                                                                                                                                                                                
 
                                                                                                                                                                                     
class TestLookupSymbol:
    def test_found_returns_fields(self, library) -> None:
        _add_element(library, 'src', 'mod.foo', description='Does the thing.')
        result = lookup_symbol(library, 'src', '/src/x.py', 'mod.foo')                                                                                                               
        assert result['found'] is True                                                                                                                                               
        assert result['qualified_name'] == 'mod.foo'                                                                                                                                 
        assert result['description'] == 'Does the thing.'                                                                                                                            
        assert result['subtype'] == 'function'
        assert result['language'] == 'python'                                                                                                                                        
 
    def test_found_without_description(self, library) -> None:                                                                                                                       
        _add_element(library, 'src', 'mod.bar')
        result = lookup_symbol(library, 'src', '/src/x.py', 'mod.bar')                                                                                                               
        assert result['found'] is True
        assert result['description'] is None                                                                                                                                         
                                                                                                                                                                                     
    def test_miss_returns_suggestions(self, library) -> None:
        # Populate with close matches in the same file and one across source                                                                                                         
        _add_element(library, 'src', 'mod.greet', file='/src/x.py')                                                                                                                  
        _add_element(library, 'src', 'mod.great', file='/src/x.py')                                                                                                                  
        _add_element(library, 'src', 'mod.totally_different', file='/src/y.py')                                                                                                      
        result = lookup_symbol(library, 'src', '/src/x.py', 'mod.greeet')                                                                                                            
        assert result['found'] is False                                                                                                                                              
        assert result['error'] == 'not_in_catalog'                                                                                                                                   
        # same-file suggestions should include greet + great                                                                                                                         
        assert 'mod.greet' in result['suggestions_in_file']                                                                                                                          
        assert 'mod.great' in result['suggestions_in_file']
                                                                                                                                                                                     
    def test_miss_no_candidates(self, library) -> None:
        result = lookup_symbol(library, 'src', '/src/x.py', 'nothing_here')                                                                                                          
        assert result['found'] is False                                                                                                                                              
        assert result['suggestions_in_file'] == []
        assert result['suggestions_in_source'] == []                                                                                                                                 
                
    def test_miss_cross_source_excluded(self, library) -> None:
        # Element with matching name but in a different source                                                                                                                       
        _add_element(library, 'other', 'mod.greet', file='/other/x.py')                                                                                                              
        result = lookup_symbol(library, 'src', '/src/x.py', 'mod.greet')                                                                                                             
        assert result['found'] is False                                                                                                                                              
        # No suggestions because 'other' source filtered out
        assert result['suggestions_in_file'] == []                                                                                                                                   
        assert result['suggestions_in_source'] == []
                                                                                                                                                                                     
                
class TestFuzzySuggestions:
    def test_separates_same_file_from_source(self, library) -> None:                                                                                                                 
        _add_element(library, 'src', 'a.greet', file='/src/a.py')
        _add_element(library, 'src', 'b.greet', file='/src/b.py')                                                                                                                    
        _add_element(library, 'src', 'c.greet', file='/src/c.py')
        in_file, in_source = fuzzy_suggestions(                                                                                                                                      
            library, 'src', '/src/a.py', 'greet',
        )                                                                                                                                                                            
        # Same-file match for a.greet
        assert 'a.greet' in in_file                                                                                                                                                  
        # Other matches land in in_source (not same file)                                                                                                                            
        for name in in_source:                                                                                                                                                       
            assert name != 'a.greet'                                                                                                                                                 
                                                                                                                                                                                     
    def test_respects_limit(self, library) -> None:
        for i in range(10):
            _add_element(library, 'src', f'mod.foo{i}', file='/src/x.py')
        in_file, _ = fuzzy_suggestions(
            library, 'src', '/src/x.py', 'mod.foo', limit=3,
        )
        assert len(in_file) <= 3

    def test_no_garbage_suggestions_from_far_names(self, library) -> None:
        # difflib over full dotted names suggested far junk (ratio 0.48 on
        # this pair); the matcher's quality floor must return nothing instead.
        _add_element(library, 'src', 'app.status.RemoteExecutor', file='/src/a.py')
        in_file, in_source = fuzzy_suggestions(library, 'src', None, 'feature_selectors')
        assert in_file == []
        assert in_source == []

    def test_bare_lowercase_name_reaches_constant_style_symbol(self, library) -> None:
        _add_element(library, 'src', 'pkg.recipes.PROBLEM_DESCRIPTION', file='/src/a.py')
        _, in_source = fuzzy_suggestions(library, 'src', None, 'problem')
        assert 'pkg.recipes.PROBLEM_DESCRIPTION' in in_source


class TestRankSymbolCandidates:
    """The suggestion matcher: segment- and word-boundary-aware with a quality
    floor — far lexical junk yields NOTHING rather than the least-bad garbage.
    """

    def test_last_segment_exact_match_ranks_first(self) -> None:
        ranked = catalog_lookup.rank_symbol_candidates(
            'TargetClass',
            ['pkg.mod.TargetClass.helper', 'pkg.mod.TargetClass', 'pkg.other.unrelated'],
        )
        assert ranked[0] == 'pkg.mod.TargetClass'
        assert 'pkg.mod.TargetClass.helper' in ranked
        assert 'pkg.other.unrelated' not in ranked

    def test_far_matches_dropped_by_quality_floor(self) -> None:
        assert catalog_lookup.rank_symbol_candidates(
            'feature_selectors', ['app.status.RemoteExecutor'],
        ) == []

    def test_token_match_case_insensitive_across_naming_styles(self) -> None:
        ranked = catalog_lookup.rank_symbol_candidates(
            'problem',
            ['pkg.recipes.PROBLEM_DESCRIPTION', 'pkg.mod.unrelated_thing'],
        )
        assert ranked == ['pkg.recipes.PROBLEM_DESCRIPTION']

    def test_near_typo_still_suggested(self) -> None:
        assert catalog_lookup.rank_symbol_candidates(
            'TargetClas', ['pkg.mod.TargetClass'],
        ) == ['pkg.mod.TargetClass']

    def test_dotted_query_matches_by_its_last_segment(self) -> None:
        assert catalog_lookup.rank_symbol_candidates(
            'mod.greeet', ['pkg.mod.greet'],
        ) == ['pkg.mod.greet']

    def test_respects_limit_deterministically(self) -> None:
        cands = [f'pkg.mod.TargetClass.m{i}' for i in range(10)]
        ranked = catalog_lookup.rank_symbol_candidates('TargetClass', cands, limit=3)
        assert ranked == [
            'pkg.mod.TargetClass.m0',
            'pkg.mod.TargetClass.m1',
            'pkg.mod.TargetClass.m2',
        ]


class TestListElementsInFile:
    def test_returns_qualified_names_for_matching_file(self, library) -> None:
        _add_element(library, 'src1', 'module.foo', file='/abs/src1/module.py')
        _add_element(library, 'src1', 'module.bar', file='/abs/src1/module.py')
        _add_element(library, 'src1', 'other.baz', file='/abs/src1/other.py')

        result = list_elements_in_file(library, 'src1', 'module.py')
        assert sorted(result) == ['module.bar', 'module.foo']

    def test_filters_out_other_sources(self, library) -> None:
        _add_element(library, 'src1', 'module.foo', file='/abs/src1/module.py')
        _add_element(library, 'src2', 'module.foo2', file='/abs/src2/module.py')

        result = list_elements_in_file(library, 'src1', 'module.py')
        assert result == ['module.foo']

    def test_filters_out_non_element_kinds(self, library) -> None:
        _add_element(library, 'src1', 'module.foo', file='/abs/src1/module.py')
        # Add a file_index doc — not an element
        library.add_document(
            content_type='catalog',
            title='file_index:src1:module.py',
            content='index',
            source_files=['/abs/src1/module.py'],
            metadata={
                'kind': 'file_index',
                'source_name': 'src1',
                'language': 'python',
            },
        )

        result = list_elements_in_file(library, 'src1', 'module.py')
        assert result == ['module.foo']

    def test_returns_empty_for_no_matches(self, library) -> None:
        _add_element(library, 'src1', 'module.foo', file='/abs/src1/module.py')
        assert list_elements_in_file(library, 'src1', 'nonexistent.py') == []

    def test_suffix_match_either_direction(self, library) -> None:
        _add_element(library, 'src1', 'module.foo', file='/long/path/to/src1/module.py')

        # Query with just the filename should match
        assert list_elements_in_file(library, 'src1', 'module.py') == ['module.foo']
        # Query with the full path should also match
        assert list_elements_in_file(
            library, 'src1', '/long/path/to/src1/module.py',
        ) == ['module.foo']


class TestCandidatePool:
    """The suggestion candidate pool must be per-source and unbounded.

    Defect: the pool was the 100k most-recently-updated catalog docs GLOBALLY
    (``list_documents(content_type='catalog', limit=100_000)``) — one big,
    recently-imported source evicted every other source's rows (and the tail
    of its own), so lookups returned found=False with zero suggestions for
    symbols that exist in the catalog.
    """

    def test_storage_lists_all_element_names_for_one_source(self, library) -> None:
        _add_element(library, 'src1', 'pkg.mod.alpha_fn', file='/abs/src1/mod.py')
        _add_element(library, 'src1', 'pkg.mod.BetaClass', file='/abs/src1/mod.py')
        _add_element(library, 'src2', 'env.pkg.gamma_fn', file='/abs/src2/m.py')
        library.add_document(  # non-element catalog doc must be excluded
            content_type='catalog',
            title='file_index:src1:mod.py',
            content='index',
            source_files=['/abs/src1/mod.py'],
            metadata={'kind': 'file_index', 'source_name': 'src1', 'language': 'python'},
        )
        pairs = library.list_catalog_element_names('src1')
        assert sorted(pairs) == [
            ('pkg.mod.BetaClass', '/abs/src1/mod.py'),
            ('pkg.mod.alpha_fn', '/abs/src1/mod.py'),
        ]

    def test_bulk_recent_source_does_not_evict_older_source(self, library) -> None:
        # Regression at the defect's real scale: >100k rows in another source,
        # all updated more recently than the target source's one element.
        _add_element(library, 'src1', 'pkg.mod.TargetClass', file='/abs/src1/mod.py')
        bulk_rows = (
            (
                f'bulk-{i}',
                f'env.pkg.bulk_{i}',
                '["/abs/src2/m.py"]',
                '2027-01-01T00:00:00+00:00',
                '2027-01-01T00:00:00+00:00',
                json.dumps({
                    'kind': 'element',
                    'source_name': 'src2',
                    'qualified_name': f'env.pkg.bulk_{i}',
                }),
            )
            for i in range(100_001)
        )
        with library._conn_provider.acquire() as conn:
            conn.executemany(
                'INSERT INTO documents '
                '(id, content_type, title, content, source_files, embedding, '
                ' created_at, updated_at, metadata, source_name) '
                "VALUES (?, 'catalog', ?, 'x', ?, NULL, ?, ?, ?, NULL)",
                bulk_rows,
            )
        in_file, in_source = fuzzy_suggestions(library, 'src1', None, 'TargetClass')
        assert 'pkg.mod.TargetClass' in in_source
        assert in_file == []
