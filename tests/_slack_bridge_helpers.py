"""Shared helpers for the slack_bridge tests (not a test module)."""

from __future__ import annotations

from slack_bridge.config import BridgeConfig

# Not a secret — bound to a name (not an inline literal) so the secret linter
# (S105/S106) doesn't flag the dummy Slack tokens these tests pass.
PLACEHOLDER = 'placeholder'


def bridge_config(*, users=frozenset(), channels=frozenset(), **kwargs) -> BridgeConfig:
    """Build a BridgeConfig with placeholder tokens for tests."""
    return BridgeConfig(
        slack_bot_token=PLACEHOLDER,
        slack_app_token=PLACEHOLDER,
        allowed_users=users,
        allowed_channels=channels,
        **kwargs,
    )
