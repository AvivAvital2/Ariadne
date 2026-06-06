"""Prompt templates for LLM-based documentation generation.

This module provides carefully crafted prompt templates for generating
different types of documentation from source code analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from attrs import frozen

if TYPE_CHECKING:
    pass

DocType = Literal['explanation', 'architecture', 'qa', 'diagram', 'catalog', 'gotcha']


@frozen
class PromptTemplate:
    """A template for generating documentation prompts."""

    name: str
    doc_type: DocType
    system_prompt: str
    user_template: str
    output_format: str


# System prompts for each documentation type
EXPLANATION_SYSTEM = '''You are a technical documentation writer specializing in Python codebases.
Your task is to create clear, comprehensive explanations of how code systems work.

Guidelines:
- Write for developers who are familiar with Python but new to this codebase
- Focus on the "how" and "why", not just the "what"
- Use concrete examples from the code when helpful
- Explain design decisions and trade-offs when apparent
- Structure content with clear headings and logical flow
- Keep explanations factual and grounded in the actual code
- Do not speculate about intentions not evident in the code
- Use code snippets sparingly and only when they add clarity'''

ARCHITECTURE_SYSTEM = '''You are a software architect documenting system designs and component relationships.
Your task is to create architecture documentation that shows how components fit together.

Guidelines:
- Focus on component relationships, data flow, and system structure
- Include Graphviz DOT diagrams (```dot) where they add clarity
- Explain the rationale behind architectural decisions when evident
- Document interfaces and integration points
- Describe extension patterns and customization hooks
- Keep the documentation maintainable and accurate to the code
- Use consistent terminology from the codebase
- Balance detail with readability'''

QA_SYSTEM = '''You are a developer advocate creating FAQ-style documentation.
Your task is to anticipate common developer questions and provide clear answers.

Guidelines:
- Focus on practical, task-oriented questions
- Answer questions concisely but completely
- Include code examples when they help understanding
- Cover both basic usage and advanced scenarios
- Address common pitfalls and gotchas
- Link conceptually related topics
- Keep answers factual and grounded in the code'''

DIAGRAM_SYSTEM = '''You are a technical diagram specialist creating Graphviz DOT diagrams for code documentation.
Your task is to create clear, informative diagrams that visualize code structure and relationships.

Guidelines:
- Use a directed graph (digraph) for flow/dependencies; an undirected graph only when direction is irrelevant
- Keep diagrams focused and not too complex
- Use consistent naming from the codebase for node labels
- Include only the most important relationships
- Add brief edge labels to clarify relationships
- Ensure the diagram is syntactically valid Graphviz DOT'''


# User prompt templates
EXPLANATION_TEMPLATE = '''Create an explanation document for the following {language_framing} code.

## Module Information
{module_info}

## Source Code
```{language_fence}
{source_code}
```

## Related Modules
{related_modules}

---

Write a comprehensive explanation covering:
1. **Overview**: What this module does and its purpose in the system
2. **Key Concepts**: Important abstractions, patterns, and data structures
3. **How It Works**: Step-by-step explanation of the main functionality
4. **Usage Examples**: How to use the main classes/functions (if applicable)
5. **Integration**: How this connects with other parts of the system

Output the documentation in Markdown format with clear section headers.'''

ARCHITECTURE_TEMPLATE = '''Create an architecture document for the following code component(s).

## Component Information
{component_info}

## Source Code
```{language_fence}
{source_code}
```

## Dependencies
{dependencies}

## Dependents
{dependents}

---

Write architecture documentation covering:
1. **Component Overview**: High-level purpose and responsibilities
2. **Design Decisions**: Key architectural choices and rationale
3. **Component Diagram**: Graphviz DOT diagram showing structure (use ```dot blocks)
4. **Interfaces**: Key APIs, protocols, and integration points
5. **Data Flow**: How data moves through the component
6. **Extension Points**: How to extend or customize this component

Output the documentation in Markdown format with clear section headers.'''

QA_TEMPLATE = '''Create Q&A documentation for the following {language_framing} code.

## Module Information
{module_info}

## Source Code
```{language_fence}
{source_code}
```

## Existing Documentation
{existing_docs}

---

Generate 5-10 Q&A pairs that developers would commonly ask about this code.

Format each Q&A as:
## Question
[Question text]

## Answer
[Answer text with code examples if helpful]

Cover questions about:
- Basic usage and getting started
- Common tasks and patterns
- Configuration and customization
- Error handling and debugging
- Best practices and pitfalls to avoid

Output in Markdown format.'''

CATALOG_SYSTEM = '''You are a technical API reference writer. Create concise function catalogs.
For each public function and method: signature, parameters with types, return type, one-line purpose.
Group by class (methods) then module-level functions. Be concise — reference catalog, not tutorial.'''

CATALOG_TEMPLATE = '''Create a function catalog for this module.

## Module
{module_info}

## Source Code
```{language_fence}
{source_code}
```

Generate a catalog listing every public function and method with:
- Full signature including type hints
- One-line description from docstring or behavior
- Import path
Group by class then standalone functions.'''

CATALOG_FORMAT = '''# module_name — Function Catalog

## ClassName

### method_name(arg1: type, arg2: type) -> return_type
One-line description.

## Functions

### function_name(arg1: type) -> return_type
One-line description.

## Import
from package.module import function_name'''


GOTCHA_SYSTEM = '''You are a code quality analyst identifying gotchas and pitfalls.
For each source file, identify non-obvious behaviors, common mistakes, and edge cases
that would trip up a developer unfamiliar with the code. Focus on:
- Thread safety issues
- Type system surprises (silent nulls, implicit conversions)
- Async/sync boundary problems
- Framework-specific quirks (attrs, polars, DuckDB)
- Performance traps
Format each gotcha with: Trigger, Affected Files, Fix, Example.'''

GOTCHA_TEMPLATE = '''Identify gotchas and pitfalls in this module.

## Module
{module_info}

## Source Code
```{language_fence}
{source_code}
```

For each gotcha found, output:
### [Short title]
**Trigger:** What causes the issue
**Affected Files:** Which files are impacted
**Fix:** How to resolve or avoid it
**Category:** thread-safety | type-system | async | framework | domain | performance
**Example:** Brief code showing the pitfall'''

GOTCHA_FORMAT = '''### [Gotcha title]
**Trigger:** ...
**Affected Files:** ...
**Fix:** ...
**Category:** ...
**Example:** ...'''


DIAGRAM_TEMPLATE = '''Create a Graphviz DOT diagram for the following code structure.

## Component Information
{component_info}

## Classes and Functions
{classes_functions}

## Relationships
{relationships}

---

Create a clear Graphviz DOT diagram that visualizes:
- Class hierarchy and relationships
- Key data flows
- Component dependencies

Use a `digraph` with:
- Nodes for the key classes / functions / components (readable labels)
- Edges for calls, dependencies, and data flow (briefly labelled)
- `rankdir=LR` (or `TB`) for readability

Output ONLY the DOT code in a ```dot block, starting with `digraph`.
Keep the diagram focused and readable (max 15-20 nodes).'''


TOPIC_SYSTEM = '''You are a technical documentation writer creating cross-cutting topic guides.
Your task is to explain how multiple source files work together as a cohesive system.

Guidelines:
- Focus on the workflow and data flow ACROSS files, not individual file details
- Explain the "big picture" — how components collaborate to achieve a goal
- Use concrete class/function names from the actual code
- Include a Graphviz DOT diagram (```dot) showing the cross-file data flow
- Write for developers who understand Python but are new to this codebase
- Keep explanations grounded in what the code actually does
- Highlight entry points, key abstractions, and extension points'''

TOPIC_TEMPLATE = '''Create a cross-cutting topic guide explaining how these files work together.

## Topic
{topic_title}

## Description
{topic_description}

## Source Files
{file_summaries}

---

Write a comprehensive topic guide covering:
1. **Overview**: What this system/workflow does and why it exists
2. **Key Components**: The main classes and functions across these files and their roles
3. **Data Flow**: How data moves through the system, from input to output (include a Graphviz DOT diagram)
4. **How It Works**: Step-by-step walkthrough of the main workflow
5. **Entry Points**: Where developers interact with this system
6. **Extension Points**: How to customize or extend the behavior

Output the documentation in Markdown format with clear section headers.'''

TOPIC_FORMAT = '''## Topic Title

### Overview
[What the system does]

### Key Components
[Classes and functions across files]

### Data Flow
```dot
[flowchart showing cross-file data flow]
```

### How It Works
[Step-by-step walkthrough]

### Entry Points
[Where to start]

### Extension Points
[How to customize]'''


# Output format specifications
EXPLANATION_FORMAT = '''## Title

### Overview
[High-level description]

### Key Concepts
[Important abstractions and patterns]

### How It Works
[Detailed explanation]

### Usage
[Code examples if applicable]

### Integration
[Connections to other components]'''

ARCHITECTURE_FORMAT = '''## Component Name

### Overview
[Component purpose]

### Design Decisions
[Key choices and rationale]

### Component Diagram
```dot
[diagram]
```

### Interfaces
[Key APIs]

### Data Flow
[Data movement]

### Extension Points
[Customization hooks]'''

QA_FORMAT = '''## Question
[Question]

## Answer
[Answer with examples]

(Repeat for each Q&A pair)'''

DIAGRAM_FORMAT = '''```dot
[Valid Graphviz DOT diagram code]
```'''


# Pre-built template instances
EXPLANATION_PROMPT = PromptTemplate(
    name='explanation',
    doc_type='explanation',
    system_prompt=EXPLANATION_SYSTEM,
    user_template=EXPLANATION_TEMPLATE,
    output_format=EXPLANATION_FORMAT,
)

ARCHITECTURE_PROMPT = PromptTemplate(
    name='architecture',
    doc_type='architecture',
    system_prompt=ARCHITECTURE_SYSTEM,
    user_template=ARCHITECTURE_TEMPLATE,
    output_format=ARCHITECTURE_FORMAT,
)

QA_PROMPT = PromptTemplate(
    name='qa',
    doc_type='qa',
    system_prompt=QA_SYSTEM,
    user_template=QA_TEMPLATE,
    output_format=QA_FORMAT,
)

DIAGRAM_PROMPT = PromptTemplate(
    name='diagram',
    doc_type='diagram',
    system_prompt=DIAGRAM_SYSTEM,
    user_template=DIAGRAM_TEMPLATE,
    output_format=DIAGRAM_FORMAT,
)

CATALOG_PROMPT = PromptTemplate(
    name='catalog',
    doc_type='catalog',
    system_prompt=CATALOG_SYSTEM,
    user_template=CATALOG_TEMPLATE,
    output_format=CATALOG_FORMAT,
)

GOTCHA_PROMPT = PromptTemplate(
    name='gotcha',
    doc_type='gotcha',
    system_prompt=GOTCHA_SYSTEM,
    user_template=GOTCHA_TEMPLATE,
    output_format=GOTCHA_FORMAT,
)


# Template registry
TEMPLATES: dict[DocType, PromptTemplate] = {
    'explanation': EXPLANATION_PROMPT,
    'architecture': ARCHITECTURE_PROMPT,
    'qa': QA_PROMPT,
    'diagram': DIAGRAM_PROMPT,
    'catalog': CATALOG_PROMPT,
    'gotcha': GOTCHA_PROMPT,
}


# ---------------------------------------------------------------------------
# Theme summarization (Themes plan, Phase 4)
# ---------------------------------------------------------------------------
#
# Cross-cutting themes are clusters discovered by Leiden over the hybrid
# graph; the LLM's job is to recognize the unifying concern (or declare
# the cluster algorithmic noise via INCOHERENT). These prompts aren't
# wired into the PromptTemplate/DocType flow because the theme path
# doesn't take source_code; callers in docgen/themes.py format them directly.

THEME_SYSTEM_PROMPT = '''\
You are a senior engineer analyzing a code cluster discovered via graph clustering.
Your job: determine WHAT THIS CLUSTER IS ABOUT (the cross-cutting theme), and produce
a clear theme document.

The cluster was found ALGORITHMICALLY by combining import structure with embedding
similarity. It is NOT necessarily a coherent theme — sometimes the algorithm groups
things that share surface features but no real concern. If that is the case, say so.

Anti-patterns to avoid in your output:
- Generic titles ("Helpers", "Utilities", "Common Code", "Misc") — these mean you
  failed to find the actual theme
- Restating member names — readers can see the list themselves
- Padding with boilerplate sections when there is nothing interesting to say
'''

THEME_USER_TEMPLATE = '''\
Analyze this cluster of {n_members} code elements. Top {k_shown} members shown by
graph centrality. Code samples included for the top members.

Members:
{member_list_with_summaries}

Sample code from anchor members:
{code_snippets}

Cross-references (impact_radius):
{impact_summaries}

Produce a markdown document with:

# <Title>
A short, specific noun phrase. Examples: "Retry Logic with Exponential Backoff",
"DB Connection Lifecycle", "LLM Prompt Construction". Forbidden: anything generic.

## What this is
2-3 sentences describing the theme.

## Why this is a coherent theme
One paragraph: what is the unifying concern? What problem do these members
collectively solve? What invariant or pattern do they share?

## Key participants
Bulleted list of the most important members with their role in the theme.
Each bullet: `- **<member>** — <role in this theme>`. List 5-15 max.

## Cross-cutting concerns
Which other parts of the codebase does this theme touch? Reference modules
or other themes if obvious. Be specific.

## Caveats
Any noise in the cluster — members that don't quite fit, ambiguity in the
theme, anything a reader should know to interpret this correctly.
Be honest. If there's no caveat, write "None apparent."

---

If the members do NOT form a coherent theme — if the cluster looks like
algorithmic noise rather than a real concept — output exactly the literal
string:

INCOHERENT

Followed by a one-paragraph explanation of why. The cluster will be skipped.
'''


def get_template(doc_type: DocType) -> PromptTemplate:
    """Get the prompt template for a documentation type.

    Args:
        doc_type: The type of documentation to generate.

    Returns:
        The corresponding PromptTemplate.

    Raises:
        KeyError: If doc_type is not recognized.
    """
    return TEMPLATES[doc_type]


def static_scaffold(doc_type: DocType) -> str:
    """The fixed prompt scaffolding for ``doc_type`` — system prompt plus
    template framing and output spec, with the ``{...}`` content slots
    left as literals.

    This is the per-call, content-independent part of the prompt the cost
    estimator can tokenize exactly (the file content and metadata are
    counted separately). Returns ``''`` for an unknown doc_type so callers
    can fall back to a flat heuristic.
    """
    template = TEMPLATES.get(doc_type)
    if template is None:
        return ''
    return '\n'.join((
        template.system_prompt,
        template.user_template,
        template.output_format,
    ))


def format_module_info(module_name: str, docstring: str | None, classes: list[str], functions: list[str]) -> str:
    """Format module information for prompt injection.

    Args:
        module_name: The module name.
        docstring: The module docstring.
        classes: List of class names.
        functions: List of function names.

    Returns:
        Formatted string for prompt.
    """
    parts = [f'**Module**: `{module_name}`']
    if docstring:
        parts.append(f'**Description**: {docstring.split(chr(10))[0]}')
    if classes:
        parts.append(f"**Classes**: {', '.join(classes)}")
    if functions:
        parts.append(f"**Functions**: {', '.join(functions)}")
    return '\n'.join(parts)


def format_related_modules(modules: list[tuple[str, str | None]]) -> str:
    """Format related module information for prompt injection.

    Args:
        modules: List of (module_name, description) tuples.

    Returns:
        Formatted string for prompt.
    """
    if not modules:
        return 'No directly related modules identified.'

    lines = []
    for name, desc in modules:
        if desc:
            lines.append(f'- `{name}`: {desc}')
        else:
            lines.append(f'- `{name}`')
    return '\n'.join(lines)


def format_cross_source_callers(
    callers: 'tuple[ScipCaller, ...]',
) -> str:
    """Render the inbound side of a file's SCIP cross-source edges.

    Each line names a remote function/method (in another file or
    source) that calls into this file. Used as the ``{dependents}``
    slot in the architecture prompt — gives the LLM ground truth
    about who depends on this file's API instead of the placeholder
    "(Analysis not performed)".
    """
    if not callers:
        return ''
    lines = []
    for c in callers:
        lines.append(
            f'- `{c.remote_qualified_name}` (in `{c.remote_source_name}`, '
            f'{c.remote_file}:{c.remote_line}) calls `{c.local_qualified_name}`'
        )
    return '\n'.join(lines)


def format_cross_source_callees(
    callees: 'tuple[ScipCallee, ...]',
) -> str:
    """Render the outbound side of a file's SCIP cross-source edges.

    Each line names a remote target this file calls. Pairs with
    ``format_cross_source_callers`` to give the LLM bidirectional
    cross-file context — distinct from import-derived dependency
    lists, which only see what was statically declared.
    """
    if not callees:
        return ''
    lines = []
    for c in callees:
        lines.append(
            f'- `{c.local_qualified_name}` calls `{c.remote_qualified_name}` '
            f'(in `{c.remote_source_name}`, {c.remote_file}:{c.remote_line})'
        )
    return '\n'.join(lines)


def format_dependencies(imports: list[str], internal: list[str]) -> str:
    """Format dependency information for prompt injection.

    Args:
        imports: List of all imported modules.
        internal: List of internal package dependencies.

    Returns:
        Formatted string for prompt.
    """
    parts = []
    if internal:
        parts.append('**Internal Dependencies**:')
        for mod in internal:
            parts.append(f'- `{mod}`')
    if imports:
        external = [i for i in imports if i not in internal]
        if external:
            parts.append('\n**External Dependencies**:')
            for mod in external[:10]:  # Limit external deps
                parts.append(f'- `{mod}`')
            if len(external) > 10:
                parts.append(f'- ... and {len(external) - 10} more')

    return '\n'.join(parts) if parts else 'No significant dependencies.'


# ---------------------------------------------------------------------------
# Per-language adaptation (Catalog transition Phase 2.4)
# ---------------------------------------------------------------------------
#
# Templates use {language_fence} and {language_framing} placeholders so the
# same prompts serve every supported language. Code-fence label adapts so
# the LLM sees `\`\`\`javascript` for a JS bundle, etc. Framing word adapts
# so the prompt says "JavaScript code" / "JSON code" / "Python code" as
# appropriate. Doc-type curation per language (e.g. only 'explanation' for
# JSON/YAML/MD) lives in LANGUAGE_DOC_TYPES.
#
# The legacy generator path passes language='python' so existing prompts
# render byte-identical to before this change.


LANGUAGE_FENCE: dict[str, str] = {
    'python': 'python',
    'html': 'html',
    'javascript': 'javascript',
    'json': 'json',
    'yaml': 'yaml',
    'markdown': 'markdown',
    'scala': 'scala',
    'java': 'java',
}

LANGUAGE_FRAMING: dict[str, str] = {
    'python': 'Python',
    'html': 'HTML',
    'javascript': 'JavaScript',
    'json': 'JSON',
    'yaml': 'YAML',
    'markdown': 'Markdown',
    'scala': 'Scala',
    'java': 'Java',
}

LANGUAGE_DOC_TYPES: dict[str, tuple[DocType, ...]] = {
    'python':     ('explanation', 'architecture', 'qa', 'catalog', 'gotcha', 'diagram'),
    'html':       ('explanation', 'architecture', 'catalog'),
    'javascript': ('explanation', 'architecture', 'qa', 'catalog', 'gotcha', 'diagram'),
    'json':       ('explanation',),
    'yaml':       ('explanation',),
    'markdown':   ('explanation',),
    # Scala/Java get the full set — implicits, traits, and OO patterns
    # benefit from architecture/qa/gotcha/diagram, just like Python.
    'scala':      ('explanation', 'architecture', 'qa', 'catalog', 'gotcha', 'diagram'),
    'java':       ('explanation', 'architecture', 'qa', 'catalog', 'gotcha', 'diagram'),
}


def render_user_template(
    template: PromptTemplate,
    *,
    language: str = 'python',
    **kwargs: object,
) -> str:
    """Format a prompt template with language-aware placeholders.

    Both the legacy ModuleMetadata-based path and the new
    EnrichedFileBundle path go through this helper. The legacy path
    always passes ``language='python'``; the new path passes
    ``bundle.language``.
    """
    fence = LANGUAGE_FENCE.get(language, language)
    framing = LANGUAGE_FRAMING.get(language, language.capitalize())
    return template.user_template.format(
        language_fence=fence,
        language_framing=framing,
        **kwargs,
    )


def filter_doc_types_for_language(
    requested: tuple[DocType, ...], language: str,
) -> tuple[DocType, ...]:
    """Intersect ``requested`` with what ``language`` supports.

    For Python/JS this is a no-op (full doc-type set). For data formats
    (JSON/YAML/MD) it filters down to ('explanation',) so we don't try
    to write architecture docs for a config dict.
    """
    allowed = LANGUAGE_DOC_TYPES.get(language)
    if allowed is None:
        return requested
    return tuple(t for t in requested if t in allowed)


def format_classes_functions(classes: list[dict], functions: list[dict]) -> str:
    """Format class and function information for diagram prompts.

    Args:
        classes: List of class info dicts with name, methods, bases.
        functions: List of function info dicts with name, args, returns.

    Returns:
        Formatted string for prompt.
    """
    parts = []

    if classes:
        parts.append('**Classes**:')
        for cls in classes:
            bases = f"({', '.join(cls.get('bases', []))})" if cls.get('bases') else ''
            parts.append(f"- `{cls['name']}{bases}`")
            for method in cls.get('methods', [])[:5]:
                parts.append(f'  - `{method}`')
            if len(cls.get('methods', [])) > 5:
                parts.append(f"  - ... and {len(cls['methods']) - 5} more methods")

    if functions:
        parts.append('\n**Functions**:')
        for func in functions:
            parts.append(f"- `{func['name']}({', '.join(func.get('args', []))})`")

    return '\n'.join(parts) if parts else 'No classes or functions found.'
