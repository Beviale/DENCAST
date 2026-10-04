"""Reading and writing Spark datasets.

Every write here is distributed: each partition is serialized by the task that
already holds it, with no data passing through the driver. That is what keeps
the usable dataset size independent of driver memory.

On Windows this requires Hadoop's native helpers -- winutils.exe for the
output committer and hadoop.dll for NativeIO -- located through HADOOP_HOME
(see `dencast.spark_session._configure_hadoop_home`). When they are missing,
the write raises rather than degrading to a driver-side collect: a write that
cannot be distributed is an environment problem to fix, and silently routing
several hundred megabytes of Python objects through the driver hides it until
the dataset outgrows the machine.
"""

from __future__ import annotations

from pathlib import Path
import shutil

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType


def write_parquet(df: DataFrame, path: Path | str, mode: str = "overwrite") -> None:
    """Write a DataFrame to Parquet: a directory holding one file per partition.

    Parquet carries its schema and row-group statistics in each file's footer,
    so the parts are independent and need no central index -- which is exactly
    what lets the tasks write them without coordinating.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write.mode(mode).parquet(str(path))
    logger.debug("Wrote {}", path)


def read_parquet(spark: SparkSession, path: Path | str) -> DataFrame:
    """Read a Parquet dataset, whether it is a directory of parts or one file."""
    return spark.read.parquet(str(path))


def write_csv(df: DataFrame, path: Path | str) -> None:
    """Write a DataFrame to a single CSV file.

    Spark always writes a directory of part-files, but these outputs are
    reports rather than datasets: DVC reads `per_day_metrics.csv` as a plot,
    and people open `anomaly_scores.csv` in a spreadsheet. So the frame is
    coalesced to one partition and the single part is moved into place.

    `coalesce(1)` does mean one task holds the whole report. That is the point
    of a report, and it is why this function is deliberately not used for
    datasets -- `write_parquet` is.

    Array columns are flattened to semicolon-separated strings, since CSV has
    no way to carry them.
    """
    flat = df
    for field in df.schema.fields:
        if isinstance(field.dataType, ArrayType):
            flat = flat.withColumn(field.name, F.concat_ws(";", F.col(field.name)))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Written beside the target rather than into it, so a failure part-way
    # leaves the previous report intact instead of a half-written directory
    # sitting where a file is expected.
    staging = path.with_name(path.name + ".spark")
    if staging.exists():
        shutil.rmtree(staging)

    flat.coalesce(1).write.mode("overwrite").option("header", "true").csv(str(staging))

    parts = sorted(staging.glob("part-*.csv"))
    if len(parts) != 1:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(
            f"Expected exactly one part file under {staging}, found {len(parts)}"
        )

    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    parts[0].replace(path)
    shutil.rmtree(staging, ignore_errors=True)
    logger.debug("Wrote {}", path)
