"""Tests for docgen.catalog_extractor."""                                                                                                                     
from __future__ import annotations

from pathlib import Path

from docgen.catalog_extractor import extract_elements


def _write(root: Path, rel: str, content: str) -> Path:
    p = root / rel                                                                                                                                                                   
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)                                                                                                                                                            
    return p    


class TestPythonExtraction:
    def test_simple_function(self, tmp_path: Path) -> None:                                                                                                                          
        f = _write(tmp_path, 'mod.py', 'def greet(name):\n    return name\n')
        els = extract_elements(f, tmp_path)                                                                                                                                          
        assert len(els) == 1
        assert els[0].subtype == 'function'                                                                                                                                          
        assert els[0].qualified_name == 'mod.greet'
        assert els[0].language == 'python'                                                                                                                                           
 
    def test_async_function(self, tmp_path: Path) -> None:                                                                                                                           
        f = _write(tmp_path, 'mod.py', 'async def work():\n    return 0\n')
        els = extract_elements(f, tmp_path)                                                                                                                                          
        assert len(els) == 1
        assert els[0].subtype == 'async_function'                                                                                                                                    
                
    def test_class_with_method(self, tmp_path: Path) -> None:
        f = _write(tmp_path, 'mod.py', 'class Car:\n    def drive(self):\n        pass\n')                                                                                        
        els = extract_elements(f, tmp_path)
        by_qn = {e.qualified_name: e.subtype for e in els}                                                                                                                           
        assert by_qn['mod.Car'] == 'class'                                                                                                                                           
        assert by_qn['mod.Car.drive'] == 'method'                                                                                                                                    
                                                                                                                                                                                     
    def test_class_attribute_has_parent(self, tmp_path: Path) -> None:
        f = _write(tmp_path, 'mod.py', 'class Car:\n    speed: int = 0\n')                                                                                                         
        els = extract_elements(f, tmp_path)                                                                                                                                          
        variables = [e for e in els if e.subtype == 'variable']                                                                                                                      
        assert len(variables) == 1                                                                                                                                                   
        assert variables[0].qualified_name == 'mod.Car.speed'
        assert variables[0].parent_qualified_name == 'mod.Car'                                                                                                                       
                                                                                                                                                                                     
    def test_submodule_qualified_name(self, tmp_path: Path) -> None:
        f = _write(tmp_path, 'pkg/sub/mod.py', 'def f(): pass\n')                                                                                                                   
        els = extract_elements(f, tmp_path)                                                                                                                                          
        assert els[0].qualified_name == 'pkg.sub.mod.f'
                                                                                                                                                                                     
    def test_init_collapses_to_package(self, tmp_path: Path) -> None:
        f = _write(tmp_path, 'pkg/__init__.py', 'def f(): pass\n')                                                                                                                  
        els = extract_elements(f, tmp_path)                                                                                                                                          
        assert els[0].qualified_name == 'pkg.f'
                                                                                                                                                                                     
    def test_body_sha_changes_with_content(self, tmp_path: Path) -> None:
        f = _write(tmp_path, 'a.py', 'def g():\n    return 1\n')                                                                                                                   
        before = extract_elements(f, tmp_path)[0].body_sha                                                                                                                           
        f.write_text('def g():\n    return 2\n')
        after = extract_elements(f, tmp_path)[0].body_sha                                                                                                                            
        assert before != after
                                                                                                                                                                                     
                
class TestHtmlExtraction:
    def test_semantic_element(self, tmp_path: Path) -> None:                                                                                                                         
        f = _write(tmp_path, 'page.html', '<section>Hi</section>')
        els = extract_elements(f, tmp_path)                                                                                                                                          
        assert len(els) == 1                                                                                                                                                         
        assert els[0].subtype == 'html_element'
        assert els[0].language == 'html'                                                                                                                                             
                
    def test_element_with_id(self, tmp_path: Path) -> None:
        f = _write(tmp_path, 'page.html', '<div id="hero">Hi</div>')                                                                                                                 
        els = extract_elements(f, tmp_path)
        assert len(els) == 1                                                                                                                                                         
        assert '#hero' in els[0].qualified_name
                                                                                                                                                                                     
    def test_ignores_non_semantic_without_attrs(self, tmp_path: Path) -> None:
        f = _write(tmp_path, 'page.html', '<div>Plain</div>')                                                                                                                        
        els = extract_elements(f, tmp_path)
        assert els == []                                                                                                                                                             
                                                                                                                                                                                     
 
class TestJavascriptExtraction:                                                                                                                                                      
    def test_function_declaration(self, tmp_path: Path) -> None:
        f = _write(tmp_path, 'app.js', 'function greet(n) { return n; }\n')
        els = extract_elements(f, tmp_path)                                                                                                                                          
        assert len(els) == 1
        assert els[0].subtype == 'js_function'                                                                                                                                       
        assert els[0].qualified_name.endswith('.greet')                                                                                                                              
 
    def test_class_declaration(self, tmp_path: Path) -> None:                                                                                                                        
        f = _write(tmp_path, 'app.js', 'class Widget {}\n')                                                                                                                         
        els = extract_elements(f, tmp_path)
        assert len(els) == 1                                                                                                                                                         
        assert els[0].subtype == 'js_class'                                                                                                                                          
 
    def test_const_export(self, tmp_path: Path) -> None:                                                                                                                             
        f = _write(tmp_path, 'app.js', 'export const Foo = 1;\n')
        els = extract_elements(f, tmp_path)                                                                                                                                          
        assert any(e.subtype == 'js_export' for e in els)
                                                                                                                                                                                     
                
class TestUnsupportedLanguage:
    def test_txt_returns_empty(self, tmp_path: Path) -> None:
        f = _write(tmp_path, 'notes.txt', 'plain text\n')
        assert extract_elements(f, tmp_path) == []

    def test_log_returns_empty(self, tmp_path: Path) -> None:
        f = _write(tmp_path, 'trace.log', 'INFO: hi\n')
        assert extract_elements(f, tmp_path) == []
