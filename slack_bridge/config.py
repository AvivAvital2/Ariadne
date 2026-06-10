from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml
from attrs import field, frozen

# The ariadne repo root, derived from THIS package's location rather than a
# hardcoded name. ``slack_bridge/`` is a subpackage of the repo, so its parent is
# the repo root — which means the directory can be renamed without breaking the
# MCP launch (``uv run --directory <ariadne_dir> ariadne mcp``).
_REPO_ROOT = Path(__file__).resolve().parents[1]


@frozen
class BridgeConfig:
    """Operational config for the Slack bridge. Holds NO secrets.

    Secrets (``OPENAI_API_KEY``/``ANTHROPIC_API_KEY``/Slack & OAuth tokens) come
    from the process environment / secret store at runtime — see
    ``designs/slack-bridge.md`` ("Secrets handling"). The token fields here are
    only the non-secret Slack token *handles* the app needs to start; in
    production they too should be injected from the environment.
    """

    slack_bot_token: str
    slack_app_token: str
    allowed_users: frozenset[str]
    allowed_channels: frozenset[str]
    ariadne_dir: Path = _REPO_ROOT
    model: str | None = None
    max_size: int = 50
    idle_ttl_seconds: float = 480.0
    turn_timeout_seconds: float = 240.0
    soft_timeout_seconds: float = 120.0
    enable_feedback: bool = False
    source_descriptions: Mapping[str, str] = field(factory=dict)
    source_aliases: Mapping[str, Sequence[str]] = field(factory=dict)

    def is_allowed(self, *, user: str, channel: str) -> bool:
        """Allow a request if its user OR its channel is allow-listed.

        Both lists empty → deny everyone (fail-closed). Channel allow-listing
        opens a whole channel; user allow-listing grants a person anywhere
        (e.g. DMs, whose channel id won't be in ``allowed_channels``).
        """
        return user in self.allowed_users or channel in self.allowed_channels

    @classmethod
    def from_env(cls, config_path: str | Path | None = None) -> BridgeConfig:
        """Build config from the environment + an operational yaml file.

        Secrets/tokens come from the environment (``SLACK_BOT_TOKEN`` /
        ``SLACK_APP_TOKEN``); everything else (allowlist, pool tuning, source
        descriptions + aliases) comes from the yaml file at ``config_path`` or
        ``$ARIADNE_SLACK_CONFIG``. ``ariadne_dir`` honours ``$ARIADNE_DIR`` but
        otherwise stays the rename-proof package-derived default.
        """
        path = config_path or os.environ.get('ARIADNE_SLACK_CONFIG')
        data: dict = {}
        if path:
            data = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
        pool = data.get('pool') or {}
        ariadne_dir_env = os.environ.get('ARIADNE_DIR')
        return cls(
            slack_bot_token=os.environ.get('SLACK_BOT_TOKEN', ''),
            slack_app_token=os.environ.get('SLACK_APP_TOKEN', ''),
            allowed_users=frozenset(data.get('allowed_users') or []),
            allowed_channels=frozenset(data.get('allowed_channels') or []),
            ariadne_dir=Path(ariadne_dir_env) if ariadne_dir_env else _REPO_ROOT,
            model=data.get('model') or os.environ.get('ARIADNE_SLACK_MODEL'),
            max_size=int(pool.get('max_size', 50)),
            idle_ttl_seconds=float(pool.get('idle_ttl_seconds', 480.0)),
            turn_timeout_seconds=float(pool.get('turn_timeout_seconds', 240.0)),
            enable_feedback=bool(data.get('enable_feedback', False)),
            source_descriptions=data.get('source_descriptions') or {},
            source_aliases=data.get('source_aliases') or {},
        )
