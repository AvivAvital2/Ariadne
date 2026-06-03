"""Read-only Slack bot bridging Slack → Claude (Agent SDK) → Ariadne (MCP).

The Claude agent does the decomposition and synthesis; Ariadne is
consulted read-only.
"""

from __future__ import annotations
