"""Tests for docgen.catalog_describer."""                                                                                                                               
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from docgen.catalog_describer import describe_element, describe_source_elements
from library import Library
from writer import LibraryWriter


@pytest.fixture                                                                                                                                                                      
def library(tmp_path: Path):
    lib = Library(tmp_path / 'describe-test.db')
    yield lib                                                                                                                                                                        
    lib.close()
                                                                                                                                                                                     
                
@pytest.fixture
def mocked_embedding(monkeypatch):
    async def fake_embed(self, text):
        return np.zeros(1536, dtype=np.float32)                                                                                                                                      
    async def fake_embed_batch(self, texts):
        return [np.zeros(1536, dtype=np.float32) for _ in texts]                                                                                                                     
    async def fake_get_client(self):                                                                                                                                                 
        return None
    async def fake_close(self):                                                                                                                                                      
        return None
    monkeypatch.setattr('embedding.EmbeddingService.embed', fake_embed)
    monkeypatch.setattr('embedding.EmbeddingService.embed_batch', fake_embed_batch)                                                                                                  
    monkeypatch.setattr('embedding.EmbeddingService._get_client', fake_get_client)
    monkeypatch.setattr('embedding.EmbeddingService.close', fake_close)                                                                                                              
                
                                                                                                                                                                                     
@pytest.fixture 
def mocked_chat(monkeypatch):                                                                                                                                                        
    async def fake_chat_complete(messages, *, model=None, **kwargs):
        prompt = messages[0]['content']                                                                                                                                              
        if 'greet' in prompt:                                                                                                                                                        
            return 'Says hello to the given name.'
        return 'A test description.'                                                                                                                                                 
    monkeypatch.setattr('docgen.catalog_describer.chat_complete', fake_chat_complete)
    return fake_chat_complete                                                                                                                                                        
 
                                                                                                                                                                                     
def _add_element_doc(library, source_name: str, qualified_name: str, *, description=None):
    meta = {
        'kind': 'element',
        'source_name': source_name,                                                                                                                                                  
        'subtype': 'function',
        'qualified_name': qualified_name,                                                                                                                                            
        'signature': 'def greet(name)',
        'location': {'line_start': 1, 'line_end': 2, 'col_start': 0, 'col_end': 10},
        'parent_qualified_name': None,                                                                                                                                               
        'sha_at_sync': 'deadbeef',
    }                                                                                                                                                                                
    if description:                                                                                                                                                                  
        meta['description'] = description
    return library.add_document(                                                                                                                                                     
        content_type='catalog',
        title=qualified_name,
        content=f'function {qualified_name} [python] /x.py:1-2 :: def greet(name)',
        source_files=['/x.py'],                                                                                                                                                      
        embedding=np.zeros(1536, dtype=np.float32),
        metadata=meta,                                                                                                                                                               
    )                                                                                                                                                                                
 
                                                                                                                                                                                     
def _run(coro): 
    return asyncio.run(coro)


class TestDescribeElement:
    def test_returns_stripped_response(self, mocked_chat) -> None:
        metadata = {                                                                                                                                                                 
            'subtype': 'function',
            'qualified_name': 'mod.greet',                                                                                                                                           
            'parent_qualified_name': None,
            'signature': 'def greet(name)',
            'file': '/x.py',                                                                                                                                                         
            'location': {'line_start': 1, 'line_end': 2},
        }                                                                                                                                                                            
        result = _run(describe_element(metadata))
        assert result == 'Says hello to the given name.'
                                                                                                                                                                                     
    def test_custom_model_passed_through(self, monkeypatch) -> None:
        captured: dict = {}                                                                                                                                                          
        async def capture(messages, *, model=None, **kwargs):
            captured['model'] = model                                                                                                                                                
            return 'desc'
        monkeypatch.setattr('docgen.catalog_describer.chat_complete', capture)                                                                                                       
        metadata = {'qualified_name': 'x.y', 'signature': 'def y()', 'location': {}}
        _run(describe_element(metadata, model='gpt-4o-mini'))                                                                                                                        
        assert captured['model'] == 'gpt-4o-mini'
                                                                                                                                                                                     
                
class TestDescribeSourceElements:                                                                                                                                                    
    def test_skips_docs_with_existing_description(
        self, library, mocked_embedding, mocked_chat,
    ) -> None:                                                                                                                                                                       
        _add_element_doc(library, 'testsrc', 'mod.greet', description='preexisting')
        async def go():                                                                                                                                                              
            async with LibraryWriter(library) as writer:
                return await describe_source_elements(library, writer, 'testsrc')                                                                                                    
        result = _run(go())
        assert result['described'] == 0                                                                                                                                              
        assert result['already_had_description'] == 1
                                                                                                                                                                                     
    def test_processes_undescribed_elements(
        self, library, mocked_embedding, mocked_chat,                                                                                                                                
    ) -> None:  
        _add_element_doc(library, 'testsrc', 'mod.greet')
        _add_element_doc(library, 'testsrc', 'mod.shout')                                                                                                                            
        async def go():
            async with LibraryWriter(library) as writer:                                                                                                                             
                return await describe_source_elements(library, writer, 'testsrc')
        result = _run(go())                                                                                                                                                          
        assert result['described'] == 2
        docs = library.list_documents(content_type='catalog', limit=100)                                                                                                             
        for d in docs:                                                                                                                                                               
            if d.metadata.get('kind') == 'element':
                assert d.metadata.get('description')                                                                                                                                 
                assert 'Description:' in d.content
                                                                                                                                                                                     
    def test_force_regenerates(
        self, library, mocked_embedding, mocked_chat,                                                                                                                                
    ) -> None:  
        _add_element_doc(library, 'testsrc', 'mod.greet', description='old desc')
        async def go():                                                                                                                                                              
            async with LibraryWriter(library) as writer:
                return await describe_source_elements(library, writer, 'testsrc', force=True)                                                                                        
        result = _run(go())                                                                                                                                                          
        assert result['described'] == 1
                                                                                                                                                                                     
    def test_filters_by_source_name(
        self, library, mocked_embedding, mocked_chat,
    ) -> None:
        _add_element_doc(library, 'src-a', 'mod.a')                                                                                                                                  
        _add_element_doc(library, 'src-b', 'mod.b')
        async def go():                                                                                                                                                              
            async with LibraryWriter(library) as writer:
                return await describe_source_elements(library, writer, 'src-a')                                                                                                      
        result = _run(go())
        assert result['described'] == 1                                                                                                                                              
        assert result['total_candidates'] == 1
                                                                                                                                                                                     
    def test_ignores_file_index_docs(
        self, library, mocked_embedding, mocked_chat,                                                                                                                                
    ) -> None:  
        _add_element_doc(library, 'testsrc', 'mod.greet')
        library.add_document(                                                                                                                                                        
            content_type='catalog',
            title='file_index:testsrc:mod.py',                                                                                                                                       
            content='file index',
            source_files=['/x.py'],
            embedding=np.zeros(1536, dtype=np.float32),                                                                                                                              
            metadata={'kind': 'file_index', 'source_name': 'testsrc'},
        )                                                                                                                                                                            
        async def go():
            async with LibraryWriter(library) as writer:                                                                                                                             
                return await describe_source_elements(library, writer, 'testsrc')
        result = _run(go())                                                                                                                                                          
        assert result['total_candidates'] == 1
                                                                                                                                                                                     
    def test_max_calls_caps_processing(
        self, library, mocked_embedding, mocked_chat,                                                                                                                                
    ) -> None:  
        for i in range(5):
            _add_element_doc(library, 'testsrc', f'mod.f{i}')
        async def go():                                                                                                                                                              
            async with LibraryWriter(library) as writer:
                return await describe_source_elements(library, writer, 'testsrc', max_calls=2)                                                                                       
        result = _run(go())                                                                                                                                                          
        assert result['described'] == 2
