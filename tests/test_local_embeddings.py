from dataclasses import replace

import numpy as np
import pytest

from docgen.llm.local import EmbeddingContractError
from embedding import reduce_local_embeddings
from model_config import QUERY_INSTRUCTION


@pytest.mark.asyncio
async def test_native_reduction_order_and_query_role(local_deployment):
    ctx, chat, embed = local_deployment
    native = np.arange(1, 4097, dtype=np.float64)
    embed.embedding_response = {
        'data': [
            {'index': 1, 'embedding': native[::-1].tolist()},
            {'index': 0, 'embedding': native.tolist()},
        ]
    }
    vectors = await ctx.embeddings.embed_documents(['first document', 'second document'])
    expected = np.arange(1, 3073, dtype=np.float64)
    expected /= np.linalg.norm(expected)
    assert len(vectors) == 2
    assert vectors[0].shape == (3072,) and vectors[0].dtype == np.float32
    np.testing.assert_allclose(vectors[0], expected.astype(np.float32))
    second = native[::-1][:3072]
    np.testing.assert_allclose(vectors[1], (second / np.linalg.norm(second)).astype(np.float32))
    assert embed.requests[-1][1]['input'] == ['first document', 'second document']
    embed.embedding_response = None
    query = await ctx.embeddings.embed_query('How does arithmetic work?')
    assert embed.requests[-1][1]['input'] == [
        f'Instruct: {QUERY_INSTRUCTION}\nQuery:How does arithmetic work?'
    ]
    assert query.nbytes == 12288
    assert chat.requests == []
    with pytest.raises(ValueError, match='explicit'):
        await ctx.embeddings.embed('ambiguous role')
    before = len(embed.requests)
    assert await ctx.embeddings.embed_documents([]) == []
    with pytest.raises(ValueError, match='indices'):
        await ctx.embeddings.embed_documents(['valid', ' '])
    assert len(embed.requests) == before
    await ctx.aclose()


@pytest.mark.parametrize(
    'mutation', ['wrong_width', 'nan', 'inf', 'zero', 'boolean', 'string', 'duplicates', 'missing']
)
def test_embedding_contract_rejections(mutation):
    rows = [{'index': i, 'embedding': [1.0] * 4096} for i in range(2)]
    if mutation == 'wrong_width':
        rows[0]['embedding'].pop()
    elif mutation in {'nan', 'inf', 'boolean', 'string'}:
        rows[0]['embedding'][0] = {
            'nan': float('nan'),
            'inf': float('inf'),
            'boolean': True,
            'string': '1',
        }[mutation]
    elif mutation == 'zero':
        rows[0]['embedding'][:3072] = [0.0] * 3072
    elif mutation == 'duplicates':
        rows[1]['index'] = 0
    else:
        rows.pop()
    with pytest.raises(EmbeddingContractError):
        reduce_local_embeddings({'data': rows}, 2)


@pytest.mark.asyncio
async def test_token_and_item_batch_limits(local_deployment):
    ctx, _, embed = local_deployment
    ctx.settings = replace(
        ctx.settings, embeddings=replace(ctx.settings.embeddings, batch_max_tokens=64, batch_size=3)
    )
    values = await ctx.embeddings.embed_documents(['a', 'b', 'c', 'd', 'e'])
    assert len(values) == 5
    batches = [
        payload['input'] for route, payload, _ in embed.requests if route.endswith('/embeddings')
    ]
    assert batches == [['a', 'b'], ['c', 'd'], ['e']]
    await ctx.aclose()
