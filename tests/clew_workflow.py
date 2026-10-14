"""Small source-backed clew fixture shared by local and hosted verification."""

from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace

QUESTION = 'How does payments.accept_payment authorize a payment and record its receipt?'
ROUTE = ['payments.accept_payment', 'payments.authorize_payment', 'payments.record_receipt']
SOURCE = '''def accept_payment(amount):
    return authorize_payment(amount)


def authorize_payment(amount):
    if amount <= 0:
        raise ValueError('Payment must be positive')
    return record_receipt(amount)


def record_receipt(amount):
    return {'receipt': amount}
'''
DISTRACTOR = '''def start_reset(token):
    return validate_token(token)


def validate_token(token):
    return consume_token(token)


def consume_token(token):
    return {'consumed': token}
'''
ROOT = Path(__file__).resolve().parents[1]


def script(name):
    return runpy.run_path(str(ROOT / 'evaluation' / 'answer-path' / f'{name}.py'))


def seed_graph(library, root, source_name):
    root.mkdir(parents=True, exist_ok=True)
    with library._conn_provider.acquire() as conn:
        for module, content, functions in (
            ('payments', SOURCE, [('accept_payment', 1, 2), ('authorize_payment', 5, 8),
                                  ('record_receipt', 11, 12)]),
            ('password', DISTRACTOR, [('start_reset', 1, 2), ('validate_token', 5, 6),
                                     ('consume_token', 9, 10)]),
        ):
            path = root / f'{module}.py'
            path.write_text(content)
            ids = [f'scip-python python fixture 1 `{module}`/{name}().' for name, _, _ in functions]
            for cid, (name, start, end) in zip(ids, functions, strict=True):
                conn.execute(
                    'INSERT INTO scip_symbols (canonical_id,source_name,language,file,'
                    'line_start,line_end,kind,display_name,qualified_name,parent_qualified_name) '
                    'VALUES (?,?,?,?,?,?,?,?,?,?)',
                    (cid, source_name, 'python', str(path), start, end, 'function', name,
                     f'{module}.{name}', ''),
                )
            for index in range(2):
                conn.execute(
                    'INSERT INTO scip_edges (caller_canonical_id,callee_canonical_id,'
                    'edge_type,file,line,confidence) VALUES (?,?,?,?,?,?)',
                    (ids[index], ids[index + 1], 'call', str(path), functions[index][2], 'exact'),
                )


def build_routes(library, source_name):
    args = SimpleNamespace(source=source_name, seeds_per_theme=200, per_anchor=1,
                           min_steps=3, dry_run=False)
    with library._conn_provider.acquire() as conn:
        return script('write_clews')['write_routes'](conn, args, library.path)


async def add_guide(library, source_name):
    from writer import LibraryWriter

    # No source-file seeds: the clew supplies the route to the compiler walk.
    return await LibraryWriter(library).add_document(
        'explanation', 'Payment receipt workflow',
        '# Payment receipt workflow\n\nThe payment API authorizes positive amounts '
        'and records a receipt. Non-positive amounts are rejected.',
        source_files=[], source_name=source_name, doc_id='clew-payment-guide',
    )


def seed_theme(library, source_name):
    from docgen.catalog_writer import _element_doc_id

    catalog = library.add_document(
        'catalog', ROUTE[0], 'Accept a payment.', source_name=source_name,
        doc_id=_element_doc_id(source_name, ROUTE[0]),
        metadata={'qualified_name': ROUTE[0], 'kind': 'element', 'language': 'python'},
    )
    theme = library.add_document('theme', 'Payment receipt workflow',
                                 'Authorize a payment and record its receipt.',
                                 source_name=source_name)
    with library._conn_provider.acquire() as conn:
        conn.execute(
            'INSERT INTO themes(cluster_id,doc_id,member_count,resolution,last_built_at,'
            'last_summarized_at,summary_hash) VALUES (?,?,1,1,?,?,?)',
            ('payments', theme.id, '2026-10-14', '2026-10-14', 'fixture'),
        )
        conn.execute('INSERT INTO theme_members(cluster_id,element_id,joined_at) VALUES (?,?,?)',
                     ('payments', catalog.id, '2026-10-14'))
