"""Frozen, source-grounded cases for remote generation and retrieval evaluation.

``source`` is the only code input for document generation. Required facts and
observations are review oracles, never prompt material. Observations are asserted
against the actual production functions in test_remote_accuracy_cases.py.

The composite Python inputs are analysis fixtures, not executable standalone
copies of Ariadne: minimal class wrappers retain original method indentation.
Every original segment and full source file carries a SHA-256 digest. A runner
must archive these snapshots before requests and reject subsequent source drift.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _segment(path: str, qualified_name: str) -> dict:
    raw = (ROOT / path).read_text()
    current = ast.parse(raw)
    parts = qualified_name.split('.')
    for part in parts:
        candidates = [
            node for node in current.body
            if getattr(node, 'name', None) == part
            or isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == part for target in node.targets)
        ]
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one source definition: {path}:{qualified_name}')
        current = candidates[0]
    decorators = getattr(current, 'decorator_list', ())
    start = min([current.lineno, *(node.lineno for node in decorators)])
    end = current.end_lineno
    text = ''.join(raw.splitlines(keepends=True)[start - 1:end])
    return {
        'id': f'{path}:{qualified_name}', 'path': path,
        'qualified_name': qualified_name,
        'start_line': start, 'end_line': end, 'text': text,
        'sha256': _sha(text), 'file_sha256': _sha(raw),
    }


def _case(case_id, split, definitions, facts, observations, questions):
    segments = [_segment(path, symbol) for path, symbol in definitions]
    scaffolding = [
        'from __future__ import annotations',
        'File/line comments and class wrappers group exact repository extracts.',
        'Omitted imported objects/helper implementations are not established by this input.',
        'Composite source is for analysis; it is not a runnable replacement module.',
    ]
    chunks = ['from __future__ import annotations\n']
    previous_class = None
    previous_path = None
    for segment in segments:
        path, name = segment['path'], segment['qualified_name']
        owner = name.split('.')[0] if '.' in name else None
        if path != previous_path or owner != previous_class:
            chunks.append(f'\n# Repository source: {path}\n')
            if owner:
                chunks.append(f'class {owner}:  # Harness wrapper for selected original methods.\n')
        prefix = '    ' if owner else ''
        chunks.append(f'{prefix}# Original lines {segment["start_line"]}-{segment["end_line"]}.\n')
        chunks.append(segment['text'] + '\n')
        previous_class, previous_path = owner, path
    source = ''.join(chunks)
    ast.parse(source)
    return {
        'case_id': case_id, 'split': split,
        'reservation_status': (
            'Reserved from model prompt tuning until candidate is frozen'
            if split == 'confirmation' else 'Development case; not a generalization claim'
        ),
        'source_name': f'remote_accuracy/{case_id}.py',
        'language': 'python', 'source': source, 'source_sha256': _sha(source),
        'source_segments': segments, 'scaffolding': scaffolding,
        'doc_types': ['explanation', 'qa'],
        'required_facts': [{'id': f'{case_id}.F{i}', 'text': text}
                           for i, text in enumerate(facts, 1)],
        'observations': observations,
        'questions': [{'id': f'{case_id}.Q{i}', 'question': question,
                       'relevant_segment_ids': [segments[n]['id'] for n in indexes]}
                      for i, (question, indexes) in enumerate(questions, 1)],
    }


def frozen_cases() -> dict[str, dict]:
    """Return independent serializable snapshots; never send review keys to a model."""
    result = {}

    def add(*args):
        case = _case(*args)
        result[case['case_id']] = case

    add(
        'config_mutation', 'development',
        [('config.py', 'Config._mutate_config'), ('config.py', 'Config.remove_source')],
        [
            'An absent source returns False without rewriting the configuration file.',
            'Removing an existing source deletes its entry and clears a matching top-level default_source.',
            'Removing a source does not remove that name from other sources depends_on lists.',
            'Only the singleton False callback result aborts; a return value of 0 still commits.',
            'A successful mutation dumps YAML, reloads through _load_from_path, and returns True.',
            'No config path and caught OSError/YAMLError return False; arbitrary callback exceptions such as RuntimeError propagate.',
            'The shown code provides no transactional rollback or atomic file replacement guarantee.',
        ],
        {
            'absent': {'returned': False, 'bytes_unchanged': True},
            'removed': {'returned': True, 'sources': ['product'], 'default_source': None,
                        'consumer_dependencies': ['shared']},
            'abort': {'returned': False, 'bytes_unchanged': True},
            'zero_return': {'returned': True, 'persisted': 'committed'},
            'callback_error': 'RuntimeError', 'no_path': False, 'invalid_yaml': False,
        },
        [('Does removing a source also rewrite the dependencies of its consumers?', [1]),
         ('Which callback results abort the configuration write and which errors propagate?', [0])],
    )
    add(
        'dependency_scope', 'development',
        [('config.py', f'Config.{name}') for name in (
            'get_source_dependencies', 'scope_closure', '_forward_closure',
            '_reverse_closure', '_reject_cycles_in_forward_deps',
        )],
        [
            'A source with declared dependencies uses forward transitive closure, including itself.',
            'A leaf with no declared dependencies uses reverse transitive closure of its consumers, including itself.',
            'A nonleaf does not automatically include sibling consumers: left depending on base has scope {left, base}.',
            'Visited sets deduplicate diamond paths and results are frozensets.',
            'An unknown requested source raises KeyError and lists configured source names.',
            'The forward path rejects a reachable dependency cycle with ValueError naming the cycle; reverse traversal has no call to that cycle checker.',
        ],
        {
            'closures': {
                'base': ['base', 'left', 'right', 'top'], 'left': ['base', 'left'],
                'right': ['base', 'right'], 'top': ['base', 'left', 'right', 'top'],
                'isolated': ['isolated'],
            },
            'unknown': 'KeyError', 'cycle': 'ValueError',
        },
        [('How does depends_on change scope for a leaf library compared with a consumer?', [1, 2, 3]),
         ('Does a consumer include sibling consumers of its dependencies?', [2]),
         ('How are unknown sources and dependency cycles reported?', [1, 4])],
    )
    add(
        'embedding_delegation', 'development',
        [('writer.py', 'LibraryWriter.update_document_embedding'),
         ('embedding.py', '_DOC_EMBED_CONTENT_CHARS'), ('embedding.py', 'doc_embedding_text')],
        [
            'A missing document returns None before obtaining the embedding service.',
            'A non-None inference context with a non-None binding uses nested_write, prepare_document and apply_prepared_document.',
            'The bound path returns the apply_prepared_document result and bypasses the legacy service.embed/update_document path.',
            'The unbound path obtains the embedding service, calls the shared doc_embedding_text helper, awaits embed, then updates only the embedding argument.',
            'doc_embedding_text preserves the full title, adds two newlines, and retains only the first 2000 characters of content; falsey content becomes empty text.',
            'The shown excerpts do not establish the service model, retry policy, vector dimensions, or the internals of prepare_document.',
        ],
        {
            'legacy': {
                'returned': 'updated-sentinel', 'input': 'A title\n\n' + 'x' * 2000,
                'update_arguments': [['doc-id', {'embedding': 'vector-sentinel'}]],
                'embedding_calls': 1,
            },
            'missing': {'returned': None, 'additional_embedding_calls': 0},
            'bound': {'returned': 'bound-sentinel',
                      'events': ['enter', 'prepare', ['apply', 'prepared-sentinel'], 'exit'],
                      'legacy_embedding_calls': 0},
        },
        [('How does an unbound writer construct document embedding text across the two modules?', [0, 1, 2]),
         ('What happens when the document is missing or the writer has a model binding?', [0])],
    )
    add(
        'catalog_metadata', 'development',
        [('docgen/catalog_describer.py', name) for name in (
            '_PROMPT', 'build_describe_prompt', 'describe_element',
        )],
        [
            'The builder formats declared metadata fields; implementation bodies and existing description values are not inserted.',
            'Missing or falsey parent metadata becomes (module-level); missing location bounds default to zero.',
            'The prompt instructs metadata-only interface description and forbids inferring runtime behavior from names; these are instructions, not a correctness guarantee.',
            'describe_element sends one user message containing the builder result and forwards the optional model argument to chat_complete.',
            'The returned chat string is stripped; this code does not validate its factual accuracy or persist the result.',
            'The prompt builder leaves the supplied metadata unchanged.',
        ],
        {'fields_present': True, 'body_absent': True, 'old_description_absent': True,
         'metadata_unchanged': True, 'default_parent': True, 'default_location': True,
         'returned': 'Returned description.', 'model_argument': 'chosen'},
        [('Can the catalog description infer runtime behavior from the supplied metadata?', [0, 1]),
         ('How are missing parent/location fields and returned whitespace handled?', [1, 2])],
    )
    add(
        'chunk_boundaries', 'confirmation',
        [('writer.py', name) for name in (
            'DEFAULT_CHUNK_SIZE', 'DEFAULT_CHUNK_OVERLAP', 'ChunkConfig',
            'chunk_text', '_split_long_text',
        )],
        [
            'Empty input returns an empty list and short nonempty input is returned as one unchanged chunk.',
            'The default chunk size and overlap are 500 and 50 characters, not tokens.',
            'Paragraph mode splits at runs of at least two newlines and does not add overlap between ordinary paragraph chunks.',
            'Long paragraphs or fixed mode use _split_long_text, preferring late punctuation and then spaces before fixed boundaries.',
            'The helper clamps excessive overlap below chunk_size and explicitly enforces forward progress.',
            'For chunk_size <= 0 the helper returns the full nonempty text as one chunk; produced ordinary slices are stripped and empty slices omitted.',
        ],
        {'empty': [], 'short': ['abc'], 'fixed_overlap': ['abcd', 'cdef', 'efgh'],
         'clamped_overlap': ['abcd', 'bcde', 'cdef'], 'nonpositive': ['abcdef'],
         'paragraphs': ['aaa', 'bbb']},
        [('What are the size units and does paragraph chunking always add overlap?', [0, 1, 2, 3]),
         ('How does fixed chunking avoid an infinite loop when overlap exceeds chunk size?', [4])],
    )
    add(
        'endpoint_routes', 'confirmation',
        [('model_config.py', name) for name in (
            'ModelConfigurationError', 'normalize_url', 'tokenize_url',
        )],
        [
            'normalize_url accepts explicit HTTP or HTTPS URLs with a hostname and rejects credentials, query strings and fragments.',
            'It removes trailing slashes from the path while retaining deployment prefixes and the original netloc spelling.',
            'tokenize_url first normalizes and then requires the normalized string to end exactly in /v1.',
            'The tokenizer endpoint replaces only the final /v1 with /tokenize, preserving preceding deployment path segments.',
            'Rejected configuration is reported with ModelConfigurationError, a ValueError subclass; these helpers perform no network health check.',
        ],
        {'normalized': 'https://EXAMPLE.COM:443/prefix/v1',
         'tokenized': 'https://host/prefix/tokenize', 'root': 'http://localhost:8080/tokenize',
         'rejected_count': 5, 'invalid_error': 'ModelConfigurationError',
         'wrong_route_error': 'ModelConfigurationError'},
        [('How is a deployment-prefixed OpenAI base URL converted to a tokenizer URL?', [1, 2]),
         ('Which endpoint URL shapes are rejected before any request?', [0, 1, 2])],
    )
    return result
