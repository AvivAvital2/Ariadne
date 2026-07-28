"""Lake jobs — Spark and Delta woven through the product (the integrated
archetype: the environment IS part of the artifact)."""
from pyspark.sql import SparkSession, functions as F


def session() -> SparkSession:
    return (
        SparkSession.builder.appName('lakerun')
        .config('spark.sql.shuffle.partitions', '64')
        .getOrCreate()
    )


def enrich_readings(spark: SparkSession, source_table: str,
                    stations_path: str, out_table: str) -> None:
    """Join the readings stream against the small stations dimension."""
    stations = spark.read.parquet(stations_path)
    readings = spark.read.table(source_table)
    enriched = readings.join(
        F.broadcast(stations), on='station_id', how='left')
    enriched.write.format('delta').mode('append').saveAsTable(out_table)


def flag_floods(spark: SparkSession, table: str) -> int:
    frame = spark.read.table(table).withColumn(
        'is_flood', F.col('level') > F.lit(4.0))
    flood_days = frame.filter('is_flood').count()
    frame.write.format('delta').mode('overwrite').saveAsTable(table)
    return flood_days
