"""Columnar SQL over local parquet files via DuckDB — the workload the
adopter's migration questions are about."""
import duckdb


def open_store(db_path: str = ':memory:') -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path)


def daily_aggregates(conn, parquet_glob: str) -> list[tuple]:
    """One local SQL pass over all parquet partitions."""
    return conn.execute(
        "SELECT station_id, date_trunc('day', taken_at) AS day, "
        'MAX(level) AS high, MIN(level) AS low '
        f"FROM read_parquet('{parquet_glob}') "
        'GROUP BY station_id, day ORDER BY day').fetchall()


def export_features(conn, parquet_glob: str, out_path: str) -> None:
    conn.execute(
        f"COPY (SELECT * FROM read_parquet('{parquet_glob}')) "
        f"TO '{out_path}' (FORMAT PARQUET)")
