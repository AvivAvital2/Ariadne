"""Delta table upkeep run after each batch of lake jobs."""
from pyspark.sql import SparkSession


def compact(spark: SparkSession, table: str) -> None:
    spark.sql(f'OPTIMIZE {table}')


def expire_old_versions(spark: SparkSession, table: str,
                        retain_hours: int = 168) -> None:
    spark.sql(f'VACUUM {table} RETAIN {retain_hours} HOURS')


def latest_writes(spark: SparkSession, table: str, limit: int = 5) -> list:
    history = spark.sql(f'DESCRIBE HISTORY {table} LIMIT {limit}')
    return [row.asDict() for row in history.collect()]
