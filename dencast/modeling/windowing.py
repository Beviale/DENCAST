"""Turning a split directory into windowed feature frames, and the Spark session."""

from __future__ import annotations

import os
import sys
from functools import reduce
from pathlib import Path
from typing import Optional, Sequence

from loguru import logger

from pyspark.sql import SparkSession

PARTS = ("train", "validation", "test")
INDEX = "datetime"


def session(app: str, workers: int = 4):
    """A local Spark session configured for this environment."""

    # The Python workers are launched by the JVM and have to find the same
    # interpreter this process runs under, or they fail to connect back.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    spark = (SparkSession.builder
             .appName(app)
             .master(f"local[{workers}]")
             .config("spark.driver.host", "127.0.0.1")
             .config("spark.driver.bindAddress", "127.0.0.1")
             .config("spark.driver.memory", "4g")
             .config("spark.sql.shuffle.partitions", "8")
             .config("spark.python.worker.reuse", "true")
             .config("spark.network.timeout", "600s")
             .config("spark.sql.execution.arrow.pyspark.enabled", "false")
             # pandas writes the index as TIMESTAMP(NANOS), which Spark 3.5 refuses
             # outright. Read as a bigint of nanoseconds and convert where needed.
             .config("spark.sql.legacy.parquet.nanosAsLong", "true")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def discover(split_dir: Path) -> tuple[str, dict[str, Path]]:
    """The dataset name and the three files."""
    found: dict[str, Path] = {}
    names = set()
    for p in PARTS:
        hits = sorted(split_dir.glob(f"*_{p}.parquet"))
        if not hits:
            raise FileNotFoundError(f"no *_{p}.parquet in {split_dir}")
        if len(hits) > 1:
            raise ValueError(f"{len(hits)} candidates for '{p}' in {split_dir}: "
                             f"{', '.join(h.name for h in hits)}")
        found[p] = hits[0]
        names.add(hits[0].name[: -len(f"_{p}.parquet")])
    if len(names) != 1:
        raise ValueError(f"the three files disagree on the dataset name: {names}")
    return names.pop(), found


def feature_columns(columns: Sequence[str]) -> list[str]:
    """Everything that is neither the index nor a label."""
    return [c for c in columns
            if c != INDEX and not c.startswith("is_") and not c.startswith("label_")]


def label_column(columns: Sequence[str]) -> str:
    """The single `is_*` column to score against."""
    flags = [c for c in columns if c.startswith("is_") or c.startswith("label_")]
    if len(flags) != 1:
        raise ValueError(f"expected exactly one 'is_*' column, found {flags}")
    return flags[0]


def _to_seconds(df, column: str = INDEX):
    """The index as seconds, whichever of the two forms Spark read it in."""
    from pyspark.sql import functions as F

    if dict(df.dtypes)[column] == "bigint":
        return F.col(column) / F.lit(1_000_000_000)
    return F.unix_timestamp(F.col(column))


def window(df, columns: list[str], seconds: int, label: str,
           categorical: list[str]):
    """Aggregate to fixed windows on absolute time.

    Numeric columns contribute the window's mean and standard deviation. Categorical
    ones contribute four summaries: the mode, the Shannon entropy, the number of distinct states and the share of the
    window spent in the dominant one.
    """
    from pyspark.sql import functions as F

    base = df.withColumn("_w", F.floor(_to_seconds(df) / seconds) * seconds)

    numeric = [c for c in columns if c not in set(categorical)]
    aggs = ([F.mean(c).alias(f"{c}_mean") for c in numeric]
            + [F.coalesce(F.stddev_pop(c), F.lit(0.0)).alias(f"{c}_std")
               for c in numeric]
            + [F.max(label).alias(label), F.count(F.lit(1)).alias("_rows")])
    out = base.groupBy("_w").agg(*aggs)
    names = [f"{c}_mean" for c in numeric] + [f"{c}_std" for c in numeric]

    if categorical:
        pair = F.explode(F.array(*[
            F.struct(F.lit(c).alias("_c"), F.col(c).cast("double").alias("_v"))
            for c in categorical]))
        counts = (base.select("_w", pair.alias("_s"))
                      .select("_w", F.col("_s._c").alias("_c"),
                              F.col("_s._v").alias("_v"))
                      .filter(F.col("_v").isNotNull())
                      .groupBy("_w", "_c", "_v").agg(F.count(F.lit(1)).alias("_n")))

  
        per = (counts.groupBy("_w", "_c")
               .agg(F.sum("_n").alias("_N"), F.max("_n").alias("_top"),
                    F.count(F.lit(1)).alias("_k"),
                    F.sum(F.col("_n") * F.log(F.col("_n"))).alias("_nlogn"),
                    F.max(F.struct(F.col("_n"), F.col("_v"))).alias("_arg"))
               .select("_w", "_c",
                       F.col("_arg._v").alias("mode"),
                       (F.log("_N") - F.col("_nlogn") / F.col("_N")).alias("entropy"),
                       F.col("_k").cast("double").alias("nunique"),
                       (F.col("_top") / F.col("_N")).alias("maxprop")))

        wide = (per.groupBy("_w").pivot("_c", categorical)
                .agg(F.first("mode").alias("mode"),
                     F.first("entropy").alias("entropy"),
                     F.first("nunique").alias("nunique"),
                     F.first("maxprop").alias("maxprop")))
        filled = [f"{c}_{k}" for c in categorical
                  for k in ("mode", "entropy", "nunique", "maxprop")]
        names += filled
        out = out.join(wide, on="_w", how="left").fillna(0.0, subset=filled)

    out = (out.withColumn(INDEX, F.timestamp_seconds("_w")).drop("_w")
              .orderBy(INDEX))
    return out, names


def prepare(spark, files: dict[str, Path], categorical: Sequence[str],
            window_seconds: int, parts: Sequence[str] = PARTS,
            features: Optional[Sequence[str]] = None) -> dict:
    """Read the parts, window them, assemble the feature vector."""
    from pyspark.ml.feature import VectorAssembler
    from pyspark.sql import functions as F

    raw = {p: spark.read.parquet(str(files[p].resolve())) for p in parts}
    first = raw[parts[0]]
    columns = feature_columns(first.columns)
    label = label_column(first.columns)
    rows = {p: raw[p].count() for p in parts}

    cat = [c for c in columns if c in set(categorical)]
    logger.info("label column `{}` | {} numeric, {} categorical: {}",
                label, len(columns) - len(cat), len(cat), ", ".join(cat) or "none")

    if window_seconds and window_seconds > 1:
        frames, built = {}, None
        for p in parts:
            frames[p], built = window(raw[p], columns, window_seconds, label, cat)
        logger.info("windowed at {} s: {} columns -> {} features (mean and std of "
                    "each numeric, mode/entropy/nunique/maxprop of each categorical)",
                    window_seconds, len(columns), len(built))
    else:
        frames = {p: raw[p].withColumn(
            INDEX, F.timestamp_seconds(_to_seconds(raw[p]))) for p in parts}
        built = columns
        logger.info("no windowing: {} features, one row per instant", len(built))

    if features is not None:
        if list(features) != list(built):
            missing = sorted(set(features) - set(built))
            extra = sorted(set(built) - set(features))
            raise ValueError(
                "the windowing does not reproduce the feature space the model was "
                f"fitted in: {len(missing)} expected columns are absent "
                f"({', '.join(missing[:5]) or 'none'}) and {len(extra)} are new "
                f"({', '.join(extra[:5]) or 'none'}); same names in a different order "
                "counts too, because a centroid's coordinates are positional")
        built = list(features)

    assembler = VectorAssembler(inputCols=list(built), outputCol="features")
    vec = {p: assembler.transform(frames[p])
                       .select(INDEX, *built, "features", label).cache()
           for p in parts}
    return {"vec": vec, "features": list(built), "label": label,
            "columns": columns, "categorical": cat, "rows": rows}


def scores(centroids: Sequence[Sequence[float]], frame,
           features: Sequence[str]):
    """Distance from each row to its nearest centroid, as a Spark column expression."""
    from pyspark.sql import functions as F

    if not centroids:
        raise ValueError("no centroids: there is nothing to measure a distance to")
    per_centre = []
    for centre in centroids:
        if len(centre) != len(features):
            raise ValueError(
                f"a centroid has {len(centre)} coordinates and the feature space has "
                f"{len(features)} columns")
        terms = [(F.col(f) - F.lit(float(v))) ** 2 for f, v in zip(features, centre)]
        per_centre.append(reduce(lambda a, b: a + b, terms))
    nearest = F.least(*per_centre) if len(per_centre) > 1 else per_centre[0]
    return F.sqrt(nearest)


__all__ = ["PARTS", "INDEX", "session", "discover", "feature_columns",
           "label_column", "window", "prepare", "scores"]
