"""Structural code catalog extractor using ast-grep.                                                                                                                    
                
For each supported source file, returns a flat list of ElementInfo                                                                                                                   
describing every meaningful code element (functions, classes, methods,                                                                                                               
module/class variables, HTML semantic elements, JS functions/classes/
exports).                                                                                                                                                                            
"""                                                                                                                                                                                  
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from ast_grep_py import SgRoot
from attrs import frozen

Language = Literal[
    'python', 'html', 'javascript', 'json', 'yaml', 'markdown',
    'scala', 'java',
    # HOCON (Typesafe Config) — file_index only; ast-grep does not
    # parse HOCON, so per-element extraction returns empty. Discovery
    # and search via the file_index doc still work.
    'hocon',
    # CSS — file_index only (no semantic symbols to extract). Closes
    # the multi-language coverage gap for products that ship CSS
    # alongside HTML / JS / TS.
    'css',
    # reStructuredText (Sphinx docs): file_index-only extraction; see designs/rst-support.md
    'rst',
]
Subtype = Literal[
    'function', 'async_function', 'class', 'method', 'variable',
    'html_element',
    'js_function', 'js_class', 'js_export', 'js_branch',
    'json_key', 'yaml_key', 'md_section',
    'hocon_key',
    'scala_class', 'scala_object', 'scala_trait', 'scala_def',
    'scala_val', 'scala_var', 'scala_implicit', 'scala_type',
    'scala_package_object',
    'java_class', 'java_interface', 'java_enum',
    'java_method', 'java_constructor', 'java_field',
    'rst_section',
]               
                                                                                                                                                                                     
_HTML_SEMANTIC = {
    'section', 'article', 'header', 'nav', 'main', 'footer',
    'aside', 'form', 'table',                                                                                                                                                        
}
                                                                                                                                                                                     
                
@frozen
class ElementInfo:
    """Structural info about a single catalog element."""
    language: Language
    subtype: Subtype
    file: str
    qualified_name: str
    signature: str
    line_start: int
    line_end: int
    col_start: int
    col_end: int
    parent_qualified_name: str | None = None
    body_sha: str = ''
    # SCIP-derived structured documentation (Scala/Java only). Dict shape
    # matches ``attrs.asdict(StructuredDoc)`` — JSON-serializable. None
    # for languages without doc enrichment (Python uses
    # PythonEnrichment.docstring instead, which lives on the EnrichedFileBundle).
    documentation: dict | None = None                                                                                                                                                               
    # Sphinx autodoc targets referenced by an rst section (rst only);
    # resolved against the SCIP index downstream. () for other languages.
    autodoc_targets: tuple[str, ...] = ()
                

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8', errors='replace')).hexdigest()                                                                                                        
 
                                                                                                                                                                                     
def _signature(node_text: str, max_lines: int = 1) -> str:
    lines = [ln.strip() for ln in node_text.splitlines() if ln.strip()]                                                                                                              
    if not lines:                                                                                                                                                                    
        return ''                                                                                                                                                                    
    return ' '.join(lines[:max_lines])                                                                                                                                               
                
def _detect_language(path: Path) -> Language | None:
    ext = path.suffix.lower()
    if ext == '.py':
        return 'python'
    if ext in ('.html', '.htm'):
        return 'html'
    if ext in ('.js', '.jsx', '.ts', '.tsx', '.mjs', '.vue'):
        # .vue routes through the javascript SCIP path; symbols come from
        # the vue-mapped index (the .vue.script.* companion's occurrences
        # translated back to .vue), never from ast-grep on raw SFC source.
        return 'javascript'
    if ext == '.json':
        return 'json'
    if ext in ('.yaml', '.yml'):
        return 'yaml'
    if ext in ('.md', '.markdown'):
        return 'markdown'
    if ext in ('.scala', '.sbt'):
        return 'scala'
    if ext == '.java':
        return 'java'
    if ext == '.conf':
        return 'hocon'
    if ext == '.css':
        return 'css'
    if ext == '.rst':
        return 'rst'
    return None
def _first_identifier(node) -> str | None:
    for child in node.children():
        if child.kind() == 'identifier':
            return child.text()                                                                                                                                                      
    return None
                                                                                                                                                                                     
                
def _py_module_qn(path: Path, source_root: Path) -> str:
    rel = path.relative_to(source_root).with_suffix('')
    parts = list(rel.parts)                                                                                                                                                          
    if parts and parts[-1] == '__init__':
        parts = parts[:-1]                                                                                                                                                           
    return '.'.join(parts) if parts else path.stem
                                                                                                                                                                                     
 
def _py_class_ancestor_chain(node, module_qn: str) -> str:                                                                                                                           
    """Return parent qualified name for a Python node (module + enclosing classes)."""                                                                                               
    names = [module_qn]                                                                                                                                                              
    for anc in reversed(list(node.ancestors())):                                                                                                                                     
        if anc.kind() == 'class_definition':                                                                                                                                         
            n = _first_identifier(anc)                                                                                                                                               
            if n:
                names.append(n)                                                                                                                                                      
    return '.'.join(names)
                                                                                                                                                                                     
 
def _py_assignment_name(node) -> str | None:                                                                                                                                         
    for child in node.children():
        kind = child.kind()
        if kind == 'identifier':
            return child.text()
        if kind == 'typed_identifier':                                                                                                                                               
            inner = _first_identifier(child)
            if inner:                                                                                                                                                                
                return inner
    text = node.text()
    head = text.split('=', 1)[0].strip().split(':', 1)[0].strip()                                                                                                                    
    return head or None                                                                                                                                                              
                                                                                                                                                                                     
                                                                                                                                                                                     
def _is_method(fn_node) -> bool:
    """True if this function_definition lives directly inside a class body."""                                                                                                       
    parent = fn_node.parent()
    if parent is None:                                                                                                                                                               
        return False
    if parent.kind() != 'block':                                                                                                                                                     
        return False
    grand = parent.parent()
    return grand is not None and grand.kind() == 'class_definition'                                                                                                                  
 
                                                                                                                                                                                     
def _extract_python(src: str, path: Path, source_root: Path) -> list[ElementInfo]:
    module_qn = _py_module_qn(path, source_root)                                                                                                                                     
    root = SgRoot(src, 'python').root()
    out: list[ElementInfo] = []                                                                                                                                                      
    file_s = str(path.resolve())                                                                                                                                                     
 
    for fn in root.find_all(kind='function_definition'):                                                                                                                             
        name = _first_identifier(fn)
        if not name:                                                                                                                                                                 
            continue
        is_async = fn.text().lstrip().startswith('async ')                                                                                                                           
        if _is_method(fn):
            subtype: Subtype = 'method'                                                                                                                                              
        elif is_async:
            subtype = 'async_function'                                                                                                                                               
        else:                                                                                                                                                                        
            subtype = 'function'
        parent_qn = _py_class_ancestor_chain(fn, module_qn)                                                                                                                          
        r = fn.range()
        out.append(ElementInfo(
            language='python', subtype=subtype, file=file_s,                                                                                                                         
            qualified_name=f'{parent_qn}.{name}',
            signature=_signature(fn.text()),                                                                                                                                         
            line_start=r.start.line + 1, line_end=r.end.line + 1,                                                                                                                    
            col_start=r.start.column, col_end=r.end.column,
            parent_qualified_name=parent_qn if parent_qn != module_qn else None,                                                                                                     
            body_sha=_sha(fn.text()),
        ))                                                                                                                                                                           
                
    for cls in root.find_all(kind='class_definition'):
        name = _first_identifier(cls)                                                                                                                                                
        if not name:
            continue                                                                                                                                                                 
        parent_qn = _py_class_ancestor_chain(cls, module_qn)
        r = cls.range()                                                                                                                                                              
        out.append(ElementInfo(
            language='python', subtype='class', file=file_s,                                                                                                                         
            qualified_name=f'{parent_qn}.{name}',
            signature=_signature(cls.text()),                                                                                                                                        
            line_start=r.start.line + 1, line_end=r.end.line + 1,
            col_start=r.start.column, col_end=r.end.column,                                                                                                                                
            parent_qualified_name=parent_qn if parent_qn != module_qn else None,                                                                                                     
            body_sha=_sha(cls.text()),
        ))                                                                                                                                                                           
                
    for asn in root.find_all(kind='assignment'):
        name = _py_assignment_name(asn)                                                                                                                                              
        if not name:
            continue                                                                                                                                                                 
        parent_qn = _py_class_ancestor_chain(asn, module_qn)
        r = asn.range()                                                                                                                                                              
        out.append(ElementInfo(
            language='python', subtype='variable', file=file_s,                                                                                                                      
            qualified_name=f'{parent_qn}.{name}',
            signature=_signature(asn.text()),                                                                                                                                        
            line_start=r.start.line + 1, line_end=r.end.line + 1,
            col_start=r.start.column, col_end=r.end.column,                                                                                                                                
            parent_qualified_name=parent_qn if parent_qn != module_qn else None,                                                                                                     
            body_sha=_sha(asn.text()),
        ))                                                                                                                                                                           
                
    return out


def _html_start_tag_name(elem) -> str:
    for c in elem.children():                                                                                                                                                        
        if c.kind() == 'start_tag':
            for cc in c.children():                                                                                                                                                  
                if cc.kind() == 'tag_name':
                    return cc.text()                                                                                                                                                 
    return ''
                                                                                                                                                                                     
                
def _html_attr(elem, attr_name: str) -> str | None:
    for attr in elem.find_all(kind='attribute'):                                                                                                                                     
        name_node = next((c for c in attr.children() if c.kind() == 'attribute_name'), None)
        if name_node and name_node.text() == attr_name:                                                                                                                              
            for vk in ('attribute_value', 'quoted_attribute_value'):
                                                                                                                              
                matches = attr.find_all(kind=vk)
                                                                                                                              
                if matches:
                                                                                                                              
                    return matches[0].text().strip('"').strip("'")
    return None                                                                                                                                                                      
                
def _extract_html(src: str, path: Path, source_root: Path) -> list[ElementInfo]:                                                                                        
    import re as _re
    module_qn = str(path.relative_to(source_root)).replace('/', '/')                                                                                                    
    root = SgRoot(src, 'html').root()                                                                                                                                   
    out: list[ElementInfo] = []                                                                                                                                         
    file_s = str(path.resolve())
    seen: set[str] = set()                                                                                                                                              
                
    for i, elem in enumerate(root.find_all(kind='element')):                                                                                                            
        tag = _html_start_tag_name(elem)
        eid = _html_attr(elem, 'id')                                                                                                                                    
        ecls = _html_attr(elem, 'class')
        if not (tag in _HTML_SEMANTIC or eid or ecls):
            continue                                                                                                                                                    
        if eid:
            ident = f'#{eid}'                                                                                                                                           
        elif ecls:
            ident = f'.{ecls.split()[0]}'
        else:                                                                                                                                                           
            ident = f'{tag}[{i}]'
        qn = f'{module_qn}{ident}'                                                                                                                                      
        if qn in seen:
            continue
        seen.add(qn)
        r = elem.range()                                                                                                                                                
        out.append(ElementInfo(
            language='html', subtype='html_element', file=file_s,                                                                                                       
            qualified_name=qn,
            signature=_signature(elem.text()),
            line_start=r.start.line + 1, line_end=r.end.line + 1,                                                                                                       
            col_start=r.start.column, col_end=r.end.column,
            parent_qualified_name=None,                                                                                                                                 
            body_sha=_sha(elem.text()),                                                                                                                                 
        ))
                                                                                                                                                                        
    # <script> elements — ast-grep tree-sitter uses kind "script_element", not "element".                                                                               
    # Descend into the inline JS and extract functions/classes with absolute line numbers.
    for script_elem in root.find_all(kind='script_element'):                                                                                                            
        r_script = script_elem.range()                                                                                                                                  
        script_text = script_elem.text()                                                                                                                                
        open_tag = _re.search(r'<script[^>]*>', script_text, _re.IGNORECASE)                                                                                            
        if not open_tag:                                                                                                                                                
            continue
        before_inner = script_text[:open_tag.end()]                                                                                                                     
        delta = before_inner.count('\n')
        close_tag = _re.search(r'</script>', script_text, _re.IGNORECASE)
        inner_end = close_tag.start() if close_tag else len(script_text)                                                                                                
        inner = script_text[open_tag.end():inner_end]
        if not inner.strip():                                                                                                                                           
            continue
        js_line_offset = r_script.start.line + delta                                                                                                                    
        out.extend(_js_elements_from_src(
            inner, module_qn, file_s, js_line_offset, parent_qn=module_qn,
        ))                                                                                                                                                              
    return out

def _js_module_qn(path: Path, source_root: Path) -> str:
    rel = path.relative_to(source_root).with_suffix('')
    return '.'.join(rel.parts) if rel.parts else path.stem                                                                                                                           
 
                                                                                                                                                                                     
def _slugify(text: str) -> str:                                                                                                                                         
    import re as _re
    s = _re.sub(r'[^A-Za-z0-9_\s-]', '', text).strip().lower()
    s = _re.sub(r'[\s-]+', '_', s)                                                                                                                                      
    return s or 'section'
                                                                                                                                                                        
                
def _extract_json(src: str, path: Path, source_root: Path) -> list[ElementInfo]:                                                                                        
    module_qn = _js_module_qn(path, source_root)
    file_s = str(path.resolve())                                                                                                                                        
    out: list[ElementInfo] = []
    try:                                                                                                                                                                
        root = SgRoot(src, 'json').root()
    except Exception:
        return out
    for obj in root.find_all(kind='object'):                                                                                                                            
        is_top = True
        parent = obj.parent()                                                                                                                                           
        while parent is not None:
            if parent.kind() in ('object', 'array', 'pair'):
                is_top = False                                                                                                                                          
                break
            parent = parent.parent()                                                                                                                                    
        if not is_top:
            continue
        for pair in obj.children():
            if pair.kind() != 'pair':
                continue                                                                                                                                                
            key = None
            for c in pair.children():                                                                                                                                   
                if c.kind() == 'string':
                    for sc in c.children():
                        if sc.kind() == 'string_content':
                            key = sc.text().strip()
                            break                                                                                                                                       
                    break
            if not key:                                                                                                                                                 
                continue
            r = pair.range()
            out.append(ElementInfo(
                language='json', subtype='json_key', file=file_s,
                qualified_name=f'{module_qn}.{key}',                                                                                                                    
                signature=_signature(pair.text()),
                line_start=r.start.line + 1, line_end=r.end.line + 1,                                                                                                   
                col_start=r.start.column, col_end=r.end.column,                                                                                                         
                body_sha=_sha(pair.text()),
                parent_qualified_name=module_qn,                                                                                                                        
            ))  
        break                                                                                                                                                           
    return out
                                                                                                                                                                        
                
def _extract_yaml(src: str, path: Path, source_root: Path) -> list[ElementInfo]:
    module_qn = _js_module_qn(path, source_root)
    file_s = str(path.resolve())                                                                                                                                        
    out: list[ElementInfo] = []
    try:                                                                                                                                                                
        root = SgRoot(src, 'yaml').root()
    except Exception:
        return out
    for mapping in root.find_all(kind='block_mapping'):                                                                                                                 
        is_top = True
        parent = mapping.parent()                                                                                                                                       
        while parent is not None:
            if parent.kind() == 'block_mapping':
                is_top = False                                                                                                                                          
                break
            parent = parent.parent()                                                                                                                                    
        if not is_top:
            continue
        for pair in mapping.children():
            if pair.kind() != 'block_mapping_pair':
                continue                                                                                                                                                
            key = None
            for c in pair.children():                                                                                                                                   
                if c.kind() in ('flow_node', 'plain_scalar'):
                    key = c.text().strip().strip("'\"")
                    break                                                                                                                                               
            if not key:
                continue                                                                                                                                                
            r = pair.range()
            out.append(ElementInfo(
                language='yaml', subtype='yaml_key', file=file_s,
                qualified_name=f'{module_qn}.{key}',                                                                                                                    
                signature=_signature(pair.text()),
                line_start=r.start.line + 1, line_end=r.end.line + 1,                                                                                                   
                col_start=r.start.column, col_end=r.end.column,                                                                                                         
                body_sha=_sha(pair.text()),
                parent_qualified_name=module_qn,                                                                                                                        
            ))  
    return out                                                                                                                                                          
 
                                                                                                                                                                        
def _extract_markdown(src: str, path: Path, source_root: Path) -> list[ElementInfo]:
    """Markdown sections keyed by heading slug.
                                                                                                                                                                        
    ast-grep has no markdown grammar today; parse with a simple line scan.
    Each ATX heading starts a section that ends at the next same-or-higher                                                                                              
    level heading or EOF. Fenced code blocks are skipped so a `#` inside                                                                                                
    code isn't mistaken for a heading.                                                                                                                                  
    """                                                                                                                                                                 
    import re as _re
    module_qn = _js_module_qn(path, source_root)
    file_s = str(path.resolve())                                                                                                                                        
    out: list[ElementInfo] = []
    lines = src.splitlines()                                                                                                                                            
    heading_re = _re.compile(r'^(#{1,6})\s+(.+?)\s*$')
    headings: list[tuple[int, int, str]] = []                                                                                                                           
    in_code_fence = False
    for idx, line in enumerate(lines):                                                                                                                                  
        if line.lstrip().startswith('```'):                                                                                                                             
            in_code_fence = not in_code_fence
            continue                                                                                                                                                    
        if in_code_fence:
            continue
        m = heading_re.match(line)
        if m:                                                                                                                                                           
            headings.append((idx, len(m.group(1)), m.group(2).strip()))
    seen: set[str] = set()                                                                                                                                              
    for i, (line_idx, level, text) in enumerate(headings):
        slug = _slugify(text)
        if slug in seen:
            j = 2
            while f'{slug}_{j}' in seen:
                j += 1
            slug = f'{slug}_{j}'                                                                                                                                        
        seen.add(slug)
        end_line = len(lines)                                                                                                                                           
        for k in range(i + 1, len(headings)):
            if headings[k][1] <= level:
                end_line = headings[k][0]                                                                                                                               
                break
        section_text = '\n'.join(lines[line_idx:end_line])                                                                                                              
        out.append(ElementInfo(
            language='markdown', subtype='md_section', file=file_s,
            qualified_name=f'{module_qn}.{slug}',                                                                                                                       
            signature=_signature(lines[line_idx]),
            line_start=line_idx + 1, line_end=end_line,                                                                                                                 
            col_start=0,                                                                                                                                                
            col_end=len(lines[line_idx]) if line_idx < len(lines) else 0,
            body_sha=_sha(section_text),                                                                                                                                
            parent_qualified_name=module_qn,                                                                                                                            
        ))
    return out                                                                                                                                                          
def _extract_rst(src: str, path: Path, source_root: Path) -> list[ElementInfo]:
    """reStructuredText sections via the docutils doctree.

    Mirrors :func:`_extract_markdown` but parses with docutils so section
    nesting and adornment styles resolve correctly. Each section becomes an
    ``rst_section`` element; nested sections link to their parent, and any
    Sphinx ``autodoc`` directives in a section are captured as
    ``autodoc_targets`` (resolved against the SCIP index downstream).
    Degrades to ``[]`` (the file-index fallback, like ``.conf``/``.css``)
    when the source has no sections or can't be parsed -- explicitly, never
    silently.
    """
    import re as _re

    from docutils import nodes as _dn
    from docutils.core import publish_doctree

    autodoc_re = _re.compile(r'^\s*\.\.\s+auto\w+::\s+(\S+)')
    module_qn = _js_module_qn(path, source_root)
    file_s = str(path.resolve())
    lines = src.splitlines()
    try:
        doctree = publish_doctree(
            src,
            settings_overrides={'report_level': 5, 'halt_level': 5, 'doctitle_xform': False},
        )
    except Exception:
        return []
    sections = list(doctree.findall(_dn.section))
    if not sections:
        return []

    # docutils reports the adornment line; the title text is the line above
    # (underline form). Fall back to line 1 if docutils gives no position.
    title_lines = [max(0, (sec.line or 1) - 2) for sec in sections]

    out: list[ElementInfo] = []
    seen: set[str] = set()
    qn_by_id: dict[int, str] = {}
    for i, sec in enumerate(sections):
        text = sec.next_node(_dn.title).astext()
        slug = _slugify(text)
        if slug in seen:
            j = 2
            while f'{slug}_{j}' in seen:
                j += 1
            slug = f'{slug}_{j}'
        seen.add(slug)
        qn = f'{module_qn}.{slug}'
        qn_by_id[id(sec)] = qn

        # Nearest enclosing section -> parent qn (else the module). Parents
        # precede children in document order, so the lookup always resolves.
        parent_qn = module_qn
        ancestor = sec.parent
        while ancestor is not None:
            if isinstance(ancestor, _dn.section):
                parent_qn = qn_by_id[id(ancestor)]
                break
            ancestor = ancestor.parent

        line_idx = title_lines[i]
        end_line = title_lines[i + 1] if i + 1 < len(sections) else len(lines)
        section_text = '\n'.join(lines[line_idx:end_line])
        targets: list[str] = []
        for ln in lines[line_idx:end_line]:
            m = autodoc_re.match(ln)
            if m:
                targets.append(m.group(1))
        out.append(ElementInfo(
            language='rst', subtype='rst_section', file=file_s,
            qualified_name=qn,
            signature=_signature(lines[line_idx]),
            line_start=line_idx + 1, line_end=end_line,
            col_start=0,
            col_end=len(lines[line_idx]),
            body_sha=_sha(section_text),
            parent_qualified_name=parent_qn,
            autodoc_targets=tuple(targets),
        ))
    return out


_SUB_EXTRACT_THRESHOLD = 80                                                                                                                                             
                                                                                                                                                                        
 
def _js_branch_name(if_node) -> str | None:                                                                                                                             
    """Derive a short branch name from the first line of an if statement.
                                                                                                                                                                        
    Matches patterns like:
      if (gateName === 'design')  -> 'design_branch'                                                                                                                    
      if (x.y === "foo")         -> 'foo_branch'                                                                                                                      
      if (someFlag)               -> 'someFlag_branch'                                                                                                                  
    """                                                                                                                                                                 
    import re as _re
    text = if_node.text().splitlines()[0]                                                                                                                               
    m = _re.search(r"===?\s*[\'\"]([A-Za-z0-9_]+)[\'\"]", text)
    if m:                                                                                                                                                               
        return m.group(1) + '_branch'
    m = _re.search(r'if\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)', text)                                                                                                      
    if m:                                                                                                                                                               
        return m.group(1) + '_branch'                                                                                                                                   
    return None                                                                                                                                                         
                                                                                                                                                                        
                
def _extract_js_sub_elements(
    fn_node, parent_qn: str, file_s: str,
line_offset: int = 0) -> list[ElementInfo]:                                                                                                                                                 
    """Extract sub-catalog entries from a large JS function body.
                                                                                                                                                                        
    Currently emits one entry per top-level if-statement branch inside the
    function, named from the guard condition. Extend with template-literals                                                                                             
    and arrow-function-named-vars as needed.                                                                                                                            
    """                                                                                                                                                                 
    out: list[ElementInfo] = []                                                                                                                                         
    seen_names: set[str] = set()                                                                                                                                        
    for if_node in fn_node.find_all(kind='if_statement'):                                                                                                               
        name = _js_branch_name(if_node)
        if not name:                                                                                                                                                    
            continue
        if name in seen_names:                                                                                                                                          
            # disambiguate duplicates (e.g. multiple "design_branch")
            idx = 2                                                                                                                                                     
            while f'{name}_{idx}' in seen_names:
                idx += 1                                                                                                                                                
            name = f'{name}_{idx}'
        seen_names.add(name)                                                                                                                                            
        r = if_node.range()
        out.append(ElementInfo(
            language='javascript',                                                                                                                                      
            subtype='js_branch',
            file=file_s,                                                                                                                                                
            qualified_name=f'{parent_qn}.{name}',
            signature=_signature(if_node.text().splitlines()[0]),                                                                                                       
            line_start=r.start.line + 1 + line_offset,
            line_end=r.end.line + 1 + line_offset,                                                                                                                                    
            col_start=r.start.column,
            col_end=r.end.column,                                                                                                                                       
            body_sha=_sha(if_node.text()),
            parent_qualified_name=parent_qn,                                                                                                                            
        ))      
    return out
def _extract_javascript(src: str, path: Path, source_root: Path) -> list[ElementInfo]:
    module_qn = _js_module_qn(path, source_root)                                                                                                                                     
    root = SgRoot(src, 'javascript').root()
    out: list[ElementInfo] = []                                                                                                                                                      
    file_s = str(path.resolve())
                                                                                                                                                                                     
    for fn in root.find_all(kind='function_declaration'):
        name = _first_identifier(fn)
        if not name:                                                                                                                                                                 
            continue
        r = fn.range()                                                                                                                                                               
        out.append(ElementInfo(
            language='javascript', subtype='js_function', file=file_s,
            qualified_name=f'{module_qn}.{name}',                                                                                                                                    
            signature=_signature(fn.text()),
            line_start=r.start.line + 1, line_end=r.end.line + 1,                                                                                                                    
            col_start=r.start.column, col_end=r.end.column,                                                                                                                                
            body_sha=_sha(fn.text()),
        ))                                                                                                                                                                           
        if (r.end.line - r.start.line) >= _SUB_EXTRACT_THRESHOLD:
            out.extend(_extract_js_sub_elements(fn, f'{module_qn}.{name}', file_s))
                
    for cls in root.find_all(kind='class_declaration'):
        name = _first_identifier(cls)                                                                                                                                                
        if not name:
            continue                                                                                                                                                                 
        r = cls.range()
        out.append(ElementInfo(
            language='javascript', subtype='js_class', file=file_s,
            qualified_name=f'{module_qn}.{name}',                                                                                                                                    
            signature=_signature(cls.text()),
            line_start=r.start.line + 1, line_end=r.end.line + 1,                                                                                                                    
            col_start=r.start.column, col_end=r.end.column,                                                                                                                                
            body_sha=_sha(cls.text()),
        ))                                                                                                                                                                           
                
    for lex in root.find_all(kind='lexical_declaration'):
        parent = lex.parent(); is_export = parent is not None and parent.kind() == 'export_statement'                                                                                                                         
        for vd in lex.find_all(kind='variable_declarator'):
            name = _first_identifier(vd)                                                                                                                                             
            if not name:
                continue                                                                                                                                                             
            r = lex.range()
            out.append(ElementInfo(
                language='javascript',                                                                                                                                               
                subtype='js_export' if is_export else 'variable',
                file=file_s, qualified_name=f'{module_qn}.{name}',                                                                                                                   
                signature=_signature(lex.text()),                                                                                                                                    
                line_start=r.start.line + 1, line_end=r.end.line + 1,
                col_start=r.start.column, col_end=r.end.column,                                                                                                                            
                body_sha=_sha(lex.text()),                                                                                                                                           
            ))
                                                                                                                                                                                     
    return out  
def _js_elements_from_src(                                                                                                                                              
    src: str,                                                                                                                                                           
    qn_prefix: str,                                                                                                                                                     
    file_s: str,                                                                                                                                                        
    line_offset: int = 0,                                                                                                                                               
    parent_qn: str | None = None,                                                                                                                                       
) -> list[ElementInfo]:
    """Extract JS function/class/lexical-declaration ElementInfos from a JS source string.                                                                              
                                                                                                                                                                        
    ``line_offset`` (0-indexed) is added to the ast-grep-reported line numbers so the
    resulting ``line_start``/``line_end`` reflect the enclosing file's absolute position.                                                                               
    Used both for standalone ``.js`` files (offset=0) and for JS embedded inside                                                                                        
    ``<script>`` blocks within HTML files.                                                                                                                              
    """                                                                                                                                                                 
    try:                                                                                                                                                                
        root = SgRoot(src, 'javascript').root()
    except Exception:
        return []                                                                                                                                                       
    out: list[ElementInfo] = []
                                                                                                                                                                        
    for fn in root.find_all(kind='function_declaration'):
        name = _first_identifier(fn)
        if not name:
            continue                                                                                                                                                    
        r = fn.range()
        out.append(ElementInfo(                                                                                                                                         
            language='javascript', subtype='js_function', file=file_s,
            qualified_name=f'{qn_prefix}.{name}',
            signature=_signature(fn.text()),                                                                                                                            
            line_start=r.start.line + 1 + line_offset,
            line_end=r.end.line + 1 + line_offset,                                                                                                                      
            col_start=r.start.column, col_end=r.end.column,                                                                                                             
            parent_qualified_name=parent_qn,
            body_sha=_sha(fn.text()),                                                                                                                                   
        ))      
        if (r.end.line - r.start.line) >= _SUB_EXTRACT_THRESHOLD:
            out.extend(_extract_js_sub_elements(fn, f'{qn_prefix}.{name}', file_s, line_offset = line_offset))
                                                                                                                                                                        
    for cls in root.find_all(kind='class_declaration'):
        name = _first_identifier(cls)                                                                                                                                   
        if not name:
            continue
        r = cls.range()
        out.append(ElementInfo(
            language='javascript', subtype='js_class', file=file_s,                                                                                                     
            qualified_name=f'{qn_prefix}.{name}',
            signature=_signature(cls.text()),                                                                                                                           
            line_start=r.start.line + 1 + line_offset,                                                                                                                  
            line_end=r.end.line + 1 + line_offset,
            col_start=r.start.column, col_end=r.end.column,                                                                                                             
            parent_qualified_name=parent_qn,                                                                                                                            
            body_sha=_sha(cls.text()),
        ))                                                                                                                                                              
                
    for lex in root.find_all(kind='lexical_declaration'):                                                                                                               
        parent = lex.parent()
        is_export = parent is not None and parent.kind() == 'export_statement'                                                                                          
        for vd in lex.find_all(kind='variable_declarator'):
            name = _first_identifier(vd)
            if not name:                                                                                                                                                
                continue
            r = lex.range()                                                                                                                                             
            out.append(ElementInfo(
                language='javascript',
                subtype='js_export' if is_export else 'variable',
                file=file_s, qualified_name=f'{qn_prefix}.{name}',                                                                                                      
                signature=_signature(lex.text()),
                line_start=r.start.line + 1 + line_offset,                                                                                                              
                line_end=r.end.line + 1 + line_offset,                                                                                                                  
                col_start=r.start.column, col_end=r.end.column,
                parent_qualified_name=parent_qn,                                                                                                                        
                body_sha=_sha(lex.text()),                                                                                                                              
            ))
                                                                                                                                                                        
    return out  
def extract_elements(
    path: Path,
    source_root: Path,
    *,
    source_config=None,
) -> list[ElementInfo]:
    """Extract all catalog elements from a source file.

    Returns an empty list for unsupported languages or unreadable files.
    Caller is responsible for filtering out dependency/virtualenv paths.

    ``source_config`` is an optional ``SourceScipConfig``: when present
    and declaring ``index_kinds.<lang> = "scip"`` for a Scala/Java file,
    routing goes through the SCIP-backed extractor instead. SCIP failures
    propagate as ``ScipError`` subclasses — callers MUST NOT silently
    fall back to ast-grep unless ``source_config.allow_degraded`` is True.
    """
    lang = _detect_language(path)
    if lang is None:
        return []

    # SCIP routing for Scala/Java when the source declares it. This is
    # checked BEFORE reading the file so a missing index fails loud
    # without ever touching ast-grep — the load-bearing tripwire.
    if lang in ('scala', 'java'):
        from docgen import scip_config as _scip_config
        index = _scip_config.resolve_index(source_config, lang)
        if index is not None:
            from docgen.scip_extractor import extract as _scip_extract
            return _scip_extract(path, source_root=source_root, index=index)
        # SCIP not declared for this language → no ast-grep grammar for
        # Scala/Java in this codebase, so return empty.
        return []

    # SCIP routing for JavaScript / TypeScript when the source declares
    # ``index_kinds.javascript = "scip"``. scip-typescript's qualified
    # names are what ``library_scip`` stores after persist; routing
    # catalog extraction through the same source aligns ElementInfo
    # qualified_names with cross-source graph entries, so Phase 2
    # Change 2's architecture-doc Dependents rendering matches symbols
    # for JS/TS files. Falls through to ast-grep when not declared so
    # un-indexed sources keep working with the structural-only path.
    if lang == 'javascript':
        from docgen import scip_config as _scip_config
        index = _scip_config.resolve_index(source_config, 'javascript')
        if index is not None:
            from docgen.scip_extractor import extract as _scip_extract
            return _scip_extract(path, source_root=source_root, index=index)
        # .vue has no ast-grep grammar (raw SFC = HTML+JS+CSS); the JS
        # grammar would emit garbage. Without a SCIP index there's nothing
        # to extract, so return empty rather than fall through.
        if path.suffix.lower() == '.vue':
            return []
        # fall through to ast-grep (plain .ts/.js still work structurally)

    try:
        src = path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError):
        return []
    if lang == 'python':
        return _extract_python(src, path, source_root)
    if lang == 'html':
        return _extract_html(src, path, source_root)
    if lang == 'javascript':
        return _extract_javascript(src, path, source_root)
    if lang == 'json':
        return _extract_json(src, path, source_root)
    if lang == 'yaml':
        return _extract_yaml(src, path, source_root)
    if lang == 'markdown':
        return _extract_markdown(src, path, source_root)
    if lang == 'rst':
        return _extract_rst(src, path, source_root)
    if lang == 'hocon':
        from docgen.hocon_extractor import _extract_hocon
        return _extract_hocon(src, path, source_root)
    return []


__all__ = ['ElementInfo', 'Language', 'Subtype', 'extract_elements']
