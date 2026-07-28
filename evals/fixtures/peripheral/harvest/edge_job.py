"""The single Spark edge: nightly ingest of raw field readings.

Everything else in harvest is Spark-free; the environment is used here
the way a peripheral consumer uses it — one job at the boundary.
"""
from pyspark.sql import SparkSession, functions as F


def ingest(readings_csv: str, out_parquet: str) -> int:
    spark = SparkSession.builder.appName('harvest-ingest').getOrCreate()
    frame = (
        spark.read.option('header', True).csv(readings_csv)
        .withColumn('tonnes', F.col('tonnes').cast('double'))
        .filter(F.col('tonnes').isNotNull())
    )
    frame.write.mode('overwrite').partitionBy('region').parquet(out_parquet)
    return frame.count()
