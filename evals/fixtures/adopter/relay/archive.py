"""Reading archive — persists fetched station readings to SQLite."""
import sqlite3


def open_archive(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        'CREATE TABLE IF NOT EXISTS readings ('
        'station_id TEXT, taken_at TEXT, level REAL, '
        'PRIMARY KEY (station_id, taken_at))')
    return conn


def store_reading(conn: sqlite3.Connection, station_id: str,
                  taken_at: str, level: float) -> None:
    conn.execute(
        'INSERT OR REPLACE INTO readings VALUES (?, ?, ?)',
        (station_id, taken_at, level))
    conn.commit()


def latest_levels(conn: sqlite3.Connection) -> dict[str, float]:
    rows = conn.execute(
        'SELECT station_id, level FROM readings r WHERE taken_at = '
        '(SELECT MAX(taken_at) FROM readings WHERE station_id = r.station_id)')
    return dict(rows.fetchall())
