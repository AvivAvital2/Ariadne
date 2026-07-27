"""Web onboarding UI for Ariadne.

A thin aiohttp backend that is itself an MCP *client*: it serves the
single-page onboarding wizard and forwards each browser request to an
Ariadne MCP tool. See ``designs/web-ui/onboarding-implementation.md``.
"""
