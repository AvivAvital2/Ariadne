"""Guard: MCP tools returning Pydantic models expose structured output schemas.

Regression test for a subtle footgun — a quoted return annotation
(``-> 'ReviewResponse'``) under ``from __future__ import annotations``, with
the model not imported at runtime, made FastMCP fail to resolve the type,
fall back to *unstructured* output, and log a "not fully defined" warning at
startup. The fix is to import the model and drop the quotes so FastMCP sees
the BaseModel and builds its schema.
"""
from __future__ import annotations


async def test_model_returning_tools_expose_structured_schemas():
    import ariadne_mcp.server as server

    tools = {t.name: t for t in await server.mcp.list_tools()}
    # These return rich Pydantic models; they must carry a structured schema.
    for name in ('ariadne_review', 'ariadne_task_context'):
        assert name in tools, f'{name} not registered'
        assert tools[name].outputSchema, (
            f'{name} has no structured outputSchema — it likely fell back to '
            'unstructured output (check for a quoted return annotation or a '
            'missing model import).'
        )
