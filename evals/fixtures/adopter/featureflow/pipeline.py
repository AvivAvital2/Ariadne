"""Local feature pipeline — single-machine, no Spark anywhere (the
adopter archetype: battery questions probe migrating this to the
environment)."""
import csv
from pathlib import Path


def load_readings(path: str) -> list[dict]:
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def compute_features(readings: list[dict]) -> list[dict]:
    """Row-wise feature derivation over the full list, in memory."""
    features = []
    for row in readings:
        level = float(row['level'])
        features.append({
            'station_id': row['station_id'],
            'level': level,
            'level_squared': level * level,
            'is_flood': level > 4.0,
        })
    return features


def run(source_csv: str, out_dir: str) -> int:
    features = compute_features(load_readings(source_csv))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out = Path(out_dir) / 'features.csv'
    with open(out, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(features[0]))
        writer.writeheader()
        writer.writerows(features)
    return len(features)
