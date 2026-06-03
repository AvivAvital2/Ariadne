"""Catalog element lookup + fuzzy suggestions.                                                                                                                   
                                                                                                                                                                                     
Fast path: direct doc-id lookup via _element_doc_id.
Miss path: fuzzy match against all catalog element qualified_names.                                                                                                                  
"""                                                                                                                                                                                  
from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Any

from docgen.catalog_writer import _element_doc_id

if TYPE_CHECKING:
    from library import Library
                

def lookup_symbol(
    library: "Library",
    source_name: str,                                                                                                                                                                
    file: str | None,
    qualified_name: str,                                                                                                                                                             
) -> dict[str, Any]:
    """Look up a single catalog element by (source, qualified_name).                                                                                                                 
                                                                                                                                                                                     
    Returns a flat dict of element fields when found:
        {found: True, language, subtype, signature, location, description,                                                                                                           
         parent_qualified_name, file, qualified_name}                                                                                                                                
                                                                                                                                                                                     
    When not found, returns:                                                                                                                                                         
        {found: False, error: "not_in_catalog",                                                                                                                                      
         suggestions_in_file: [...], suggestions_in_source: [...]}                                                                                                                   
                                                                                                                                                                                     
    The `file` parameter is used only to separate same-file suggestions from                                                                                                         
    across-source suggestions when the lookup misses. It is NOT part of the                                                                                                          
    lookup key (qualified_name is globally unique within a source).                                                                                                                  
    """                                                                                                                                                                              
    doc_id = _element_doc_id(source_name, qualified_name)                                                                                                                            
    doc = library.get_document(doc_id)                                                                                                                                               
    if doc is not None and doc.metadata.get('kind') == 'element':
        loc = doc.metadata.get('location') or {}                                                                                                                                     
        return {
            'found': True,                                                                                                                                                           
            'language': doc.metadata.get('language'),
            'subtype': doc.metadata.get('subtype'),                                                                                                                                  
            'qualified_name': doc.metadata.get('qualified_name'),
            'signature': doc.metadata.get('signature'),                                                                                                                              
            'location': {                                                                                                                                                            
                'line_start': loc.get('line_start'),
                'line_end': loc.get('line_end'),                                                                                                                                     
                'col_start': loc.get('col_start'),
                'col_end': loc.get('col_end'),                                                                                                                                       
            },
            'parent_qualified_name': doc.metadata.get('parent_qualified_name'),                                                                                                      
            'description': doc.metadata.get('description'),                                                                                                                          
            'file': (doc.source_files[0] if doc.source_files else None),
        }                                                                                                                                                                            
                
    # Miss — compute fuzzy suggestions                                                                                                                                               
    in_file, in_source = fuzzy_suggestions(library, source_name, file, qualified_name)                                                                                               
    return {
        'found': False,                                                                                                                                                              
        'error': 'not_in_catalog',                                                                                                                                                   
        'qualified_name': qualified_name,
        'suggestions_in_file': in_file,                                                                                                                                              
        'suggestions_in_source': in_source,
    }                                                                                                                                                                                
 
                                                                                                                                                                                     
def fuzzy_suggestions(
    library: "Library",
    source_name: str,
    file: str | None,
    qualified_name: str,
    limit: int = 5,
) -> tuple[list[str], list[str]]:                                                                                                                                                    
    """Return (suggestions_in_file, suggestions_in_source) — up to `limit` each.
                                                                                                                                                                                     
    Same-file matches are prioritized; cross-source matches fill the second list
    excluding any already in the first. Uses difflib for lexical similarity.                                                                                                         
    """                                                                                                                                                                              
    all_catalog = library.list_documents(content_type='catalog', limit=100_000)                                                                                                      
    same_file_qns: list[str] = []                                                                                                                                                    
    other_qns: list[str] = []
    for d in all_catalog:                                                                                                                                                            
        if d.metadata.get('source_name') != source_name:
            continue                                                                                                                                                                 
        if d.metadata.get('kind') != 'element':
            continue                                                                                                                                                                 
        qn = d.metadata.get('qualified_name') or ''
        if not qn:                                                                                                                                                                   
            continue
        d_file = d.source_files[0] if d.source_files else None                                                                                                                       
        if file is not None and d_file == file:
            same_file_qns.append(qn)                                                                                                                                                 
        else:                                                                                                                                                                        
            other_qns.append(qn)
                                                                                                                                                                                     
    in_file = difflib.get_close_matches(qualified_name, same_file_qns, n=limit, cutoff=0.4)
    seen = set(in_file)                                                                                                                                                              
    extra = [q for q in difflib.get_close_matches(
        qualified_name, other_qns, n=limit * 3, cutoff=0.4,                                                                                                                          
    ) if q not in seen][:limit]                                                                                                                                                      
    return list(in_file), extra                                                                                                                                                      
                                                                                                                                                                                     
                
def list_elements_in_file(
    library: "Library",
    source_name: str,
    file: str,
) -> list[str]:
    """List qualified_names of all catalog elements whose source file matches `file`.

    Path matching is suffix-based: the doc's source_files entry must either
    end with `file` or vice versa, mirroring `lookup_symbol`'s file scoping.
    """
    all_catalog = library.list_documents(content_type='catalog', limit=100_000)
    result: list[str] = []
    for d in all_catalog:
        if d.metadata.get('source_name') != source_name:
            continue
        if d.metadata.get('kind') != 'element':
            continue
        qn = d.metadata.get('qualified_name') or ''
        if not qn:
            continue
        d_file = d.source_files[0] if d.source_files else ''
        if not d_file:
            continue
        if d_file.endswith(file) or file.endswith(d_file):
            result.append(qn)
    return result


__all__ = ['fuzzy_suggestions', 'get_element_body', 'list_elements_in_file', 'lookup_symbol']


def get_element_body(
    library: "Library",
    source_name: str,
    file: str | None,
    qualified_name: str,
) -> dict[str, Any]:
    """Return catalog element metadata plus the current body text."""
    from pathlib import Path as _P
    info = lookup_symbol(library, source_name, file, qualified_name)
    if not info.get('found'):
        return info
    file_path = info.get('file')
    loc = info.get('location') or {}
    ls = loc.get('line_start')
    le = loc.get('line_end')
    if not file_path or ls is None or le is None:
        info['body_error'] = 'missing_file_or_location'
        return info
    try:
        text = _P(file_path).read_text(encoding='utf-8')
    except OSError as e:
        info['body_error'] = f'read_failed: {e}'
        return info
    src_lines = text.splitlines()
    body_lines = src_lines[ls - 1:le]
    info['body'] = '\n'.join(body_lines)
    info['body_line_count'] = len(body_lines)
    return info
