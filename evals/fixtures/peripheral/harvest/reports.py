"""Yield report builder — the consumer's OWN domain (no Spark here).

The peripheral archetype: most of the codebase is its own concern; Spark
appears only at the ingest edge (edge_job.py) while the project's prose
mentions the environment constantly — the vocabulary-saturation trap the
battery's control rows probe.
"""
from statistics import mean


def field_summary(rows: list[dict]) -> dict:
    yields = [float(r['tonnes']) for r in rows]
    return {
        'fields': len(rows),
        'total': round(sum(yields), 1),
        'mean': round(mean(yields), 2),
        'best_field': max(rows, key=lambda r: float(r['tonnes']))['field_id'],
    }


def render(region: str, summary: dict) -> str:
    return (f'{region}: {summary["total"]}t over {summary["fields"]} fields '
            f'(mean {summary["mean"]}t, best {summary["best_field"]})')


def pick_regions(rows: list[dict], minimum_fields: int = 3) -> list[str]:
    """Regions with enough reporting fields to publish."""
    by_region: dict[str, int] = {}
    for row in rows:
        by_region[row['region']] = by_region.get(row['region'], 0) + 1
    return sorted(r for r, n in by_region.items() if n >= minimum_fields)
