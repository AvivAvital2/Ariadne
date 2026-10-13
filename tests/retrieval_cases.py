"""Source-grounded questions authored before either embedding benchmark is run.

The versioned JSON fixture freezes excerpts, labels and supporting quotations.
Judgments are agent-authored, not independent human relevance assessments.
"""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/model_support/retrieval-corpus-v1.json'

# (document ID, repository file, heading or lead, questions with evidence anchors)
CASES = [
    (
        'source-config',
        'docs/directory-scoping.md',
        '## Configuration Reference',
        [
            (
                'Does declaring a parent require listing it again as a dependency?',
                'The parent is automatically added as an implicit dependency',
            ),
            (
                'How can I restrict a documentation source to feature branches?',
                'Git branch patterns where source is active',
            ),
            (
                "Are a source's explicitly declared dependencies followed recursively?",
                'Dependencies are loaded transitively.',
            ),
            (
                'Which source configuration field fixes an external library to a particular tag?',
                'Pin an external dependency to a specific git ref',
            ),
            (
                'Can a source declare a parent outside its directory ancestry?',
                "The child's path must be a subdirectory of the parent's path.",
            ),
        ],
    ),
    (
        'scope-resolution',
        'docs/directory-scoping.md',
        '## How It Works',
        [
            (
                'If several configured source directories contain my working directory, which one wins?',
                'Returns the most specific match',
            ),
            (
                'Which Python matching function interprets branch patterns?',
                "Branch patterns use Python's `fnmatch` module",
            ),
        ],
    ),
    (
        'scope-conflicts',
        'docs/directory-scoping.md',
        '## Conflict Resolution',
        [
            (
                'How are equally scoped competing documents ordered when their modification times differ?',
                '**Most recently updated wins**',
            ),
            (
                'How do I explicitly declare that a document replaces an older one?',
                'supersedes: old-caching-design',
            ),
        ],
    ),
    (
        'scope-commands',
        'docs/directory-scoping.md',
        '## CLI Commands',
        [
            (
                'Which manifest option disables filtering by git branch?',
                '`--no-branch-filter` - Disable branch filtering entirely',
            ),
            (
                'How do I search documentation across every source regardless of the current scope?',
                'ariadne search "caching" --include-all',
            ),
        ],
    ),
    (
        'onboard-source',
        'docs/new-project-onboarding.md',
        '## Step 1: Configure the Source',
        [
            (
                'Should a newly configured source path point to exported docs or to source code?',
                'The `path` should point to your source code root.',
            ),
        ],
    ),
    (
        'generate-docs',
        'docs/new-project-onboarding.md',
        '## Step 2: Generate Documentation',
        [
            (
                'Which generation option prints detailed validation failures?',
                'Use `--verbose` to see detailed validation output',
            ),
        ],
    ),
    (
        'integration-setup',
        'docs/new-project-onboarding.md',
        '## Step 3: Initialize the Integration',
        [
            (
                'What project-level configuration file is created when MCP registration is not global?',
                'This creates a `.mcp.json` in the target project',
            ),
            (
                'Does manually registering the MCP server install the session hooks as well?',
                "With manual registration, you won't get session hooks",
            ),
        ],
    ),
    (
        'mcp-permissions',
        'docs/new-project-onboarding.md',
        '## Step 4: Grant MCP Permissions (Optional)',
        [
            (
                'Where can a project allow read-only MCP tools without an approval for every call?',
                'add them to `.claude/settings.local.json`',
            ),
            (
                'Which permissions example allows reporting a useful search result?',
                'mcp__ariadne__ariadne_log_hit',
            ),
        ],
    ),
    (
        'docs-freshness',
        'docs/new-project-onboarding.md',
        '## Keeping Docs Fresh',
        [
            (
                'What command previews documentation that needs updating after code changes?',
                'ariadne sync --status --source myproject',
            ),
        ],
    ),
    (
        'local-onboard',
        'docs/new-project-onboarding.md',
        '## Local model onboarding',
        [
            (
                'What changes when local onboarding is run with approval?',
                '`--approve` binds the pair and executes structural/catalog work',
            ),
        ],
    ),
    (
        'local-profile',
        'docs/local-models.md',
        'lead',
        [
            (
                'Can rebuilding or forcing generation change the model pair of an existing local library?',
                '`--force`, import and rebuild cannot migrate models.',
            ),
            (
                'How are native Qwen embedding vectors reduced before storage?',
                'Ariadne stores the normalized first 3072 components.',
            ),
            (
                'What text is used to embed a complete document?',
                'Documents retain the title plus the first 2000 content characters',
            ),
            (
                'Which embedding inputs receive the retrieval instruction?',
                'Only search queries receive the fixed retrieval instruction.',
            ),
            (
                'How much extra context headroom does local generation reserve beyond its requested output?',
                'complete requested output budget plus 64 tokens',
            ),
        ],
    ),
    (
        'local-jobs',
        'docs/local-models.md',
        '## Durable local runs',
        [
            (
                'Will resuming an interrupted embedding stage request another copy of the saved chat response?',
                'Saved chat responses and parsed results survive',
            ),
            (
                'Is interrupting a local batch equivalent to explicitly cancelling it?',
                'Ctrl-C and process termination leave work resumable.',
            ),
            (
                'What distinguishes staleness records for identical filenames from different sources?',
                'Sidecar keys include source name, generation scope and relative path.',
            ),
            (
                'Can another process cancel work while the foreground worker holds the corpus write lock?',
                'Cancellation uses a short control-only transaction',
            ),
        ],
    ),
    (
        'container-layout',
        'docs/docker.md',
        'lead',
        [
            (
                'Which user ID and filesystem locations does the application container use?',
                'It runs as UID 1000, stores state under `/data`',
            ),
            (
                'Are language-specific SCIP indexers supplied in the application image?',
                'Language-specific SCIP indexers/build toolchains are not included',
            ),
        ],
    ),
    (
        'container-cloud',
        'docs/docker.md',
        '## Cloud suite',
        [
            (
                'How can I point the cloud Compose client at a different repository?',
                'set `ARIADNE_SOURCE_DIR` to an absolute path',
            ),
            (
                'Where does Compose retain database files and exports after the application container is removed?',
                'The named `ariadne-state` volume persists',
            ),
            (
                'Should I edit the bundled cloud YAML inside the running Compose container?',
                'The mounted YAML is read-only; edit configuration on the host.',
            ),
        ],
    ),
    (
        'container-local',
        'docs/docker.md',
        '## Local suite',
        [
            (
                'Does launching the application container also launch vLLM and download weights?',
                'This application image does not contain weights or start vLLM.',
            ),
            (
                'Which internal hostnames are in the local Compose client configuration?',
                '`chat:8000` and `embeddings:8000`',
            ),
        ],
    ),
    (
        'container-distribution',
        'docs/docker.md',
        '## Reproduce local distribution verification',
        [
            (
                'How does the distribution test establish that the consumer really downloaded the image?',
                'consumer starts with an empty image store',
            ),
            (
                'Does the registry verification mount the host Docker socket into its test daemons?',
                'no host Docker socket or credentials are mounted',
            ),
        ],
    ),
    (
        'mcp-branches',
        'docs/claude-code-integration.md',
        '## Branch-Aware Filtering',
        [
            (
                'What metadata marks generated material as experimental for session filtering?',
                'status: experimental',
            ),
            (
                'When will a session hook include an experimental document?',
                'only include experimental docs when working on matching branches',
            ),
        ],
    ),
    (
        'mcp-instructions',
        'docs/claude-code-integration.md',
        '## How MCP Instructions Work',
        [
            (
                'How is the directive to query Ariadne before exploring code delivered without a session hook?',
                'built-in `instructions` field',
            ),
        ],
    ),
    (
        'mcp-feedback',
        'docs/claude-code-integration.md',
        '## Usage Tracking & Feedback',
        [
            ('Does a negative result report require an explanation?', 'feedback required'),
            (
                'How do tool results expose the identifier needed for later feedback?',
                '[Usage event: <id>]',
            ),
            (
                'Which configuration controls whether the session manifest asks Claude to mention Ariadne?',
                '`mention_ariadne` configuration option',
            ),
        ],
    ),
    (
        'scip-setup',
        'README.md',
        '### Setup workflow',
        [
            (
                'Where does source discovery write the per-machine indexer plan?',
                '`<source>/.ariadne/manifest.json`',
            ),
            (
                'Does frequent synchronization invoke language indexers again?',
                'reuses the persisted graph — no scip-X invocation',
            ),
            (
                'What happens if synchronization finds a changed file in a previously undeclared language?',
                'it triggers `discover --config-only` automatically',
            ),
        ],
    ),
    (
        'vue-indexing',
        'README.md',
        '### Vue-aware TypeScript indexing',
        [
            (
                'Where does the Vue extraction helper obtain its matching component compiler?',
                "indexed project's own `node_modules`",
            ),
            (
                'How are indexed positions mapped back from generated Vue script companions?',
                'line-mapping is preserved',
            ),
        ],
    ),
    (
        'graph-queries',
        'README.md',
        '### Querying the cross-source graph',
        [
            (
                'What happens when an abbreviated symbol name identifies multiple candidates?',
                'returns a disambiguation list and exits non-zero',
            ),
            (
                'How can I bound cross-source impact analysis to three hops?',
                'impact_radius LicenseService.validate_token --depth 3',
            ),
        ],
    ),
]


def excerpt(path, heading):
    """Capture an existing Markdown section, ignoring headings inside code fences."""
    lines = path.read_text().splitlines(keepends=True)
    if heading == 'lead':
        end = next(i for i, line in enumerate(lines) if line.startswith('## '))
        return ''.join(lines[:end]).strip().replace('# ', '## ', 1), 1, end
    start = next(i for i, line in enumerate(lines) if line.rstrip() == heading)
    depth = len(heading.split(' ', 1)[0])
    fence, end = False, len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].lstrip().startswith('```'):
            fence = not fence
        match = re.match(r'^(#{1,6}) ', lines[i])
        if not fence and match and len(match[1]) <= depth:
            end = i
            break
    text = ''.join(lines[start:end]).strip()
    if depth == 3:
        text = text.replace('### ', '## ', 1)
    return text, start + 1, end


def freeze():
    """Explicit one-time snapshot; live runs never regenerate their input corpus."""
    from writer import chunk_text, split_into_sections

    documents, questions = [], []
    for doc_id, filename, heading, judgments in CASES:
        text, first, last = excerpt(ROOT / filename, heading)
        title = text.splitlines()[0].lstrip('# ') + ' / ' + Path(filename).stem
        content = f'# {title}\n\n{text}\n'
        chunks = chunk_text(content) if len(content) > 500 else []
        sections = split_into_sections(content)
        documents.append(
            {
                'id': doc_id,
                'title': title,
                'content': content,
                'source_file': filename,
                'source_lines': [first, last],
                'source_sha256': hashlib.sha256((ROOT / filename).read_bytes()).hexdigest(),
                'chunks': chunks,
                'sections': [list(s) for s in sections],
            }
        )
        for question, quote in judgments:
            relevant_chunks = [f'{doc_id}:{i}' for i, c in enumerate(chunks) if quote in c]
            relevant_sections = [f'{doc_id}:{i}' for i, (_, s) in enumerate(sections) if quote in s]
            assert quote in content and relevant_sections, (doc_id, quote)
            assert not chunks or relevant_chunks, (doc_id, quote)
            questions.append(
                {
                    'id': f'q{len(questions) + 1:02}',
                    'question': question,
                    'evidence': quote,
                    'documents': [doc_id],
                    'chunks': relevant_chunks,
                    'sections': relevant_sections,
                }
            )
    assert len(questions) == 50 and len(documents) == 22
    data = {
        'version': 1,
        'judgment_method': 'Agent-authored known-relevant evidence targets, frozen before model calls; not independent human review or exhaustive relevance labeling.',
        'corpus_method': 'Repository Markdown excerpts, with a document title prepended and the excerpt heading normalized to level two. Prose unchanged.',
        'documents': documents,
        'questions': questions,
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE.open('x') as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write('\n')


if __name__ == '__main__':
    freeze()
