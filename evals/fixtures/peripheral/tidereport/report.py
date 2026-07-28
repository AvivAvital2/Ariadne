"""Tide report builder — the consumer's OWN domain (no HTTP here).

The peripheral archetype: most of the codebase is its own concern; the
environment (httpx) appears only at the edges (edge_fetch.py), while the
project's prose mentions it constantly — the vocabulary-saturation trap
the battery's control rows probe.
"""
from statistics import mean


def daily_summary(readings: list[tuple[str, float]]) -> dict:
    """Collapse (hour, level) readings into the day's headline numbers."""
    levels = [level for _, level in readings]
    return {
        'high': max(levels),
        'low': min(levels),
        'mean': round(mean(levels), 2),
        'range': round(max(levels) - min(levels), 2),
    }


def render_text(station: str, summary: dict) -> str:
    return (f'{station}: high {summary["high"]}m, low {summary["low"]}m '
            f'(mean {summary["mean"]}m, range {summary["range"]}m)')


def schedule_key(station: str, day: str) -> str:
    """Nightly job key — one summary per station per day."""
    return f'{day}:{station}'
