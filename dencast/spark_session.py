"""Spark session creation and shared SQL helpers."""

import os
from pathlib import Path
import sys
from typing import Optional

from loguru import logger
from pyspark.sql import Column, SparkSession
from pyspark.sql import functions as F

# Imported for the side effect: it loads .env, where HADOOP_HOME lives.
import dencast.config  # noqa: F401
from dencast.utils import norm


def _configure_hadoop_home() -> None:
    """Put %HADOOP_HOME%/bin on PATH so the JVM can load hadoop.dll.
    """
    hadoop_home = os.environ.get("HADOOP_HOME")
    if not hadoop_home:
        return
    bin_dir = Path(hadoop_home) / "bin"
    if not (bin_dir / "winutils.exe").exists():
        return
    current = os.environ.get("PATH", "")
    if str(bin_dir) not in current:
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{current}"


def get_spark(params=None, app_name: str = "DENCAST") -> SparkSession:
    """Create (or retrieve) the SparkSession from the project parameters.
    """
    if params is not None:
        cfg = params.spark
        master: Optional[str] = cfg.master
        driver_memory = cfg.driver_memory
        shuffle_partitions = cfg.num_partitions
    else:
        master, driver_memory, shuffle_partitions = "local[*]", "4g", 8

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    if master and master.startswith("local"):
        os.environ.setdefault(
            "PYSPARK_SUBMIT_ARGS", f"--driver-memory {driver_memory} pyspark-shell"
        )

    if master and master.startswith("local"):
        os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")
        os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

    _configure_hadoop_home()

    builder = SparkSession.builder.appName(app_name)
    if master:
        builder = builder.master(master)
        if master.startswith("local"):
            builder = builder.config("spark.driver.host", "localhost").config(
                "spark.driver.bindAddress", "127.0.0.1"
            )
    spark = (
        builder.config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.adaptive.enabled", "true")
        # Broadcast joins off. Spark decides to broadcast from an *estimate* of
        # a side's size, and in this pipeline those estimates are unreliable:
        # the LSH stage joins a table against itself after a window function and
        # a union of num_permutations copies, so the statistics Catalyst has are
        # for the source parquet, not for what the plan actually produces.
        # Guessing low means building the whole side in the driver's heap, which
        # is 3 GB here and fails with notEnoughMemoryToBuildAndBroadcastTable.
        # The joins that a broadcast would genuinely help -- the per-cluster
        # moments, a few hundred rows -- are cheap as shuffle joins anyway.
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .config("spark.sql.maxPlanStringLength", "2048")
        .config("spark.sql.debug.maxToStringFields", "32")
        .getOrCreate()
    )
    _silence_plan_truncation_warnings(spark)
    return spark


def _silence_plan_truncation_warnings(spark: SparkSession) -> None:
    """Stop Spark warning every time it shortens a rendered query plan.

    The two caps above are set deliberately low, so these warnings fire on
    nearly every query and report the thing we asked for. They also arrive in
    bulk: an LSH plan renders to megabytes and each rendering logs a line, so
    a single stage can bury its own output under dozens of them.

    Only the two loggers that emit those messages are raised to ERROR, so a
    genuine error from the same classes still comes through, and every other
    Spark warning is untouched. The caps themselves stay -- they are what keeps
    the driver from running out of heap while building a plan string.
    """
    try:
        jvm = spark.sparkContext._jvm
        configurator = jvm.org.apache.logging.log4j.core.config.Configurator
        error = jvm.org.apache.logging.log4j.Level.ERROR
        for name in (
            "org.apache.spark.sql.catalyst.util.StringUtils",
            "org.apache.spark.sql.catalyst.util.SparkStringUtils",
        ):
            configurator.setLevel(name, error)
    except Exception as exc:  # noqa: BLE001
        # Log4j internals are not part of Spark's public API and have moved
        # between versions. This is cosmetic, so a failure here must never be
        # able to stop a session from starting -- unlike the writers in
        # dencast.io, where swallowing an error would hide a real one.
        logger.debug("Could not silence the plan-truncation warnings: {}", exc)


def dot(a: Column, b: Column) -> Column:
    """Dot product of two array<double> columns, in native Spark SQL.

    Uses zip_with + aggregate instead of a Python UDF: no serialization to the
    Python interpreter, everything stays inside Catalyst.
    """
    products = F.zip_with(a, b, lambda x, y: x * y)
    return F.aggregate(products, F.lit(0.0), lambda acc, x: acc + x)


def cosine_similarity(a: Column, b: Column) -> Column:
    """Cosine similarity between two array<double> columns.

    Returns 0.0 when either vector has zero norm, to avoid NaN.

    Note: the original Scala code has a typo here (`dotProduct(v2, v2)` instead
    of `dotProduct(v1, v2)`), which computes the ratio of the norms rather than
    the cosine. This implementation follows the paper.
    """
    denom = norm(a) * norm(b)
    return F.when(denom > 0, dot(a, b) / denom).otherwise(F.lit(0.0))
