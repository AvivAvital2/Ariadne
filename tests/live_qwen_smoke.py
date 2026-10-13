"""Compatibility entry point for the shared live model suite (defaults to Qwen)."""

from tests.live_model_smoke import main
from tests.model_smoke_cases import CASES

__all__ = ['CASES', 'main']

if __name__ == '__main__':
    raise SystemExit(main())
