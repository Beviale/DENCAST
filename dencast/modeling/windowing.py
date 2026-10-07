"""Turning a split directory into windowed feature frames. Activate the Spark session."""

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
             .config("spark.sql.codegen.maxFields", "30")
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


import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    DoubleType,
    LongType
)
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F

def window(df, columns: list[str], seconds: int, label: str, categorical: list[str]):
    """Aggregate to fixed windows on absolute time using PySpark applyInPandas.

    Numeric columns contribute the window's mean and standard deviation. Categorical
    ones contribute four summaries: the mode, the Shannon entropy, the number of distinct states,
    and the share of the window spent in the dominant one.
    
    """
    numeric = [c for c in columns if c not in set(categorical)]
    
    base = df.withColumn("_w", F.floor(_to_seconds(df) / seconds) * seconds)

    schema_fields = [
        StructField("_w", LongType(), True),
        StructField(label, DoubleType(), True),
        StructField("_rows", LongType(), True)
    ]
    
    names = []
    for c in numeric:
        mean_col = f"{c}_mean"
        std_col = f"{c}_std"
        names.extend([mean_col, std_col])
        schema_fields.append(StructField(mean_col, DoubleType(), True))
        schema_fields.append(StructField(std_col, DoubleType(), True))
        
    if categorical:
        filled = [f"{c}_{k}" for c in categorical for k in ("mode", "entropy", "nunique", "maxprop")]
        names.extend(filled)
        for col_name in filled:
            schema_fields.append(StructField(col_name, DoubleType(), True))
            
    out_schema = StructType(schema_fields)

    def process_window(key: tuple, pdf: pd.DataFrame) -> pd.DataFrame:
        w_val = key[0]
        row_count = len(pdf)
        
        row_dict = {
            "_w": w_val,
            label: float(pdf[label].max()) if label in pdf else 0.0,
            "_rows": row_count
        }
        
        for c in numeric:
            vals = pdf[c].dropna()
            if len(vals) > 0:
                row_dict[f"{c}_mean"] = float(vals.mean())
                std_val = float(vals.std(ddof=0))
                row_dict[f"{c}_std"] = 0.0 if np.isnan(std_val) else std_val
            else:
                row_dict[f"{c}_mean"] = 0.0
                row_dict[f"{c}_std"] = 0.0
                logger.warning(f"The numeric column {c} does not contain any value!")

        if categorical:
            for c in categorical:
                s = pdf[c].dropna()
                if len(s) == 0:
                    logger.warning(f"The categorical column {c} does not contain any value!")
                    row_dict[f"{c}_mode"] = 0.0
                    row_dict[f"{c}_entropy"] = 0.0
                    row_dict[f"{c}_nunique"] = 0.0
                    row_dict[f"{c}_maxprop"] = 0.0
                else:
                    counts = s.value_counts()
                    n_total = counts.sum()
                    
                    row_dict[f"{c}_mode"] = float(counts.idxmax())
                    row_dict[f"{c}_nunique"] = float(len(counts))
                    row_dict[f"{c}_maxprop"] = float(counts.max() / n_total)
                    
                    probs = counts / n_total
                    entropy_val = -float((probs * np.log(probs)).sum())
                    if np.isnan(entropy_val):                     
                        logger.warning(f"The calculated entropy is not valid!")
                    row_dict[f"{c}_entropy"] = 0.0 if np.isnan(entropy_val) else entropy_val

        return pd.DataFrame([row_dict])

    out = base.groupBy("_w").applyInPandas(process_window, schema=out_schema)

    out = out.withColumn(INDEX, F.timestamp_seconds("_w")).drop("_w")

    return out, names


def prepare(spark, files: dict[str, Path], categorical: Sequence[str],
            window_seconds: int, parts: Sequence[str] = PARTS,
            features: Optional[Sequence[str]] = None) -> dict:
    """Read the parts, window them, assemble the feature vector."""

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
            logger.info(f"Creating the windows for the {p} split...")
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
                       .select(INDEX, "features", label).cache()
           for p in parts}
    return {"vec": vec, "features": list(built), "label": label,
            "columns": columns, "rows": rows}



__all__ = ["PARTS", "INDEX", "session", "discover", "feature_columns",
           "label_column", "window", "prepare"]
