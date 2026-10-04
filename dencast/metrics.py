"""Regression metrics.

The paper evaluates DENCAST with RMSE, and with the average RMSE over the time
series in the multi-target setting. The quality of the clusters is judged
indirectly, through the accuracy of the predictions they support.
"""

from typing import Dict, List, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def _squared_errors(df: DataFrame) -> DataFrame:
    """Add per-object RMSE and MAE, averaged over the k targets.

    Expects columns `prediction` and `actual`, both array<double>.
    """
    diffs = F.zip_with(F.col("prediction"), F.col("actual"), lambda p, a: p - a)
    squared = F.transform(diffs, lambda d: d * d)
    absolute = F.transform(diffs, lambda d: F.abs(d))
    n = F.size(F.col("actual")).cast("double")

    return df.withColumn(
        "rmse",
        F.sqrt(F.aggregate(squared, F.lit(0.0), lambda acc, x: acc + x) / n),
    ).withColumn(
        "mae",
        F.aggregate(absolute, F.lit(0.0), lambda acc, x: acc + x) / n,
    )


def evaluate(df: DataFrame) -> Dict[str, float]:
    """Overall RMSE and MAE over all objects and all targets.

    RMSE is computed on the pooled squared errors rather than by averaging the
    per-object RMSEs, which is the stricter and more common convention.
    """
    diffs = F.zip_with(F.col("prediction"), F.col("actual"), lambda p, a: p - a)
    squared = F.transform(diffs, lambda d: d * d)
    absolute = F.transform(diffs, lambda d: F.abs(d))

    exploded = df.select(
        F.explode(squared).alias("se"),
    )
    exploded_abs = df.select(F.explode(absolute).alias("ae"))

    mse = exploded.agg(F.avg("se").alias("mse")).collect()[0]["mse"]
    mae = exploded_abs.agg(F.avg("ae").alias("mae")).collect()[0]["mae"]

    return {
        "rmse": float(mse) ** 0.5 if mse is not None else float("nan"),
        "mae": float(mae) if mae is not None else float("nan"),
        "n_objects": df.count(),
    }


def evaluate_paper_style(
    df: DataFrame, group_cols: Optional[list] = None
) -> Dict[str, float]:
    """RMSE aggregated the way the original implementation does it.

    The Scala code computes one RMSE per (plant, day) group and then takes a
    plain average of those RMSEs, rather than pooling all the squared errors:

        predictionsByDay.groupByKey().map(calculateErrors)
        totalRMSE = dailyErrors.map(_._4).reduce(_+_) / dailyErrors.count()

    This is not the same statistic as the pooled RMSE, and the difference is
    not a rounding detail. Because the square root is concave, the mean of the
    per-group RMSEs is at most the pooled RMSE, with equality only when every
    group has the same error. On heterogeneous groups -- a plant that is easy
    to predict and one that is hard -- the macro-average comes out visibly
    lower. Any comparison with the numbers published in the paper has to use
    this version.

    Args:
        df: needs `prediction`, `actual` and the grouping columns.
        group_cols: the grouping keys, e.g. ["group", "date"]. Falling back to
            a single group reduces this to the ordinary pooled RMSE.
    """
    diffs = F.zip_with(F.col("prediction"), F.col("actual"), lambda p, a: p - a)
    squared = F.transform(diffs, lambda d: d * d)
    absolute = F.transform(diffs, lambda d: F.abs(d))

    keys = group_cols or []
    per_row = df.select(
        *[F.col(c) for c in keys],
        F.aggregate(squared, F.lit(0.0), lambda acc, x: acc + x).alias("sse"),
        F.aggregate(absolute, F.lit(0.0), lambda acc, x: acc + x).alias("sae"),
        F.size(F.col("actual")).cast("double").alias("k"),
    )

    if keys:
        per_group = per_row.groupBy(*keys).agg(
            F.sqrt(F.sum("sse") / F.sum("k")).alias("rmse"),
            (F.sum("sae") / F.sum("k")).alias("mae"),
        )
    else:
        per_group = per_row.agg(
            F.sqrt(F.sum("sse") / F.sum("k")).alias("rmse"),
            (F.sum("sae") / F.sum("k")).alias("mae"),
        )

    agg = per_group.agg(
        F.avg("rmse").alias("rmse"),
        F.avg("mae").alias("mae"),
        F.count("*").alias("n_groups"),
    ).collect()[0]

    return {
        "rmse": float(agg["rmse"]),
        "mae": float(agg["mae"]),
        "n_groups": int(agg["n_groups"]),
    }


def evaluate_by_group(
    df: DataFrame, group_col: str
) -> DataFrame:
    """Per-object RMSE/MAE aggregated by a grouping column.

    The paper reports the error per plant and per day before averaging, which
    gives every group the same weight regardless of how many rows it holds.
    """
    with_errors = _squared_errors(df)
    return (
        with_errors.groupBy(group_col)
        .agg(
            F.avg("rmse").alias("rmse"),
            F.avg("mae").alias("mae"),
            F.count("*").alias("n"),
        )
        .orderBy(group_col)
    )


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """One-line human-readable summary."""
    return (
        f"{prefix}RMSE = {metrics['rmse']:.4f}  |  "
        f"MAE = {metrics['mae']:.4f}  |  "
        f"objects = {metrics['n_objects']}"
    )


def evaluate_regression(df: DataFrame) -> Dict[str, float]:
    """RMSE and MAE per target, plus the averages used to select a model.

    No grouping by plant. The Scala code averaged an RMSE per (plant, day),
    which needs a plant column that not every dataset has, and which makes the
    number depend on how the rows happen to be grouped rather than on how well
    the model predicts.

    `rmse_avg` is the RMSE of each target computed separately and then averaged
    over targets. That is the paper's multi-target convention, and it is the
    one worth selecting on: pooling the targets instead lets whichever target
    has the widest range dominate, so a model could win by being good at the
    large-scale target and poor at every other. With a single target the two
    coincide.

    Args:
        df: needs `prediction` and `actual`, both array<double> of length k.

    Returns:
        rmse_avg, mae_avg, rmse_pooled, mae_pooled, rmse_target_<i>, n_objects.
    """
    diffs = F.zip_with(F.col("prediction"), F.col("actual"), lambda p, a: p - a)
    # One posexplode, then both errors derived from it. Exploding the squared
    # and the absolute errors separately would cross-join the two arrays.
    per_target = df.select(F.posexplode(diffs).alias("t_idx", "d")).select(
        "t_idx",
        (F.col("d") * F.col("d")).alias("se"),
        F.abs(F.col("d")).alias("ae"),
    )

    rows = (
        per_target.groupBy("t_idx")
        .agg(F.avg("se").alias("mse"), F.avg("ae").alias("mae"), F.count("*").alias("n"))
        .orderBy("t_idx")
        .collect()
    )
    if not rows:
        return {"rmse_avg": float("nan"), "mae_avg": float("nan"), "n_objects": 0}

    rmses = [float(r["mse"]) ** 0.5 for r in rows]
    maes = [float(r["mae"]) for r in rows]
    total_n = sum(int(r["n"]) for r in rows)
    pooled_mse = sum(float(r["mse"]) * int(r["n"]) for r in rows) / total_n
    pooled_mae = sum(float(r["mae"]) * int(r["n"]) for r in rows) / total_n

    out: Dict[str, float] = {
        "rmse_avg": sum(rmses) / len(rmses),
        "mae_avg": sum(maes) / len(maes),
        "rmse_pooled": pooled_mse**0.5,
        "mae_pooled": pooled_mae,
        "n_targets": float(len(rows)),
        "n_objects": float(total_n / len(rows)),
    }
    if len(rows) > 1:
        for i, (rmse, mae) in enumerate(zip(rmses, maes)):
            out[f"rmse_target_{i}"] = rmse
            out[f"mae_target_{i}"] = mae
    return out


def regression_sums(df: DataFrame) -> Dict[int, tuple]:
    """Per-target (sum of squared errors, sum of absolute errors, count).

    The pieces `evaluate_regression` builds its averages from, exposed so a
    caller that has to work in chunks can add them up and get exactly the
    figure a single pass would have produced. Averaging per-chunk RMSEs would
    not: the square root does not commute with the mean, and chunks of
    different size would be weighted wrongly on top of that.
    """
    diffs = F.zip_with(F.col("prediction"), F.col("actual"), lambda p, a: p - a)
    rows = (
        df.select(F.posexplode(diffs).alias("t_idx", "d"))
        .groupBy("t_idx")
        .agg(
            F.sum(F.col("d") * F.col("d")).alias("se"),
            F.sum(F.abs(F.col("d"))).alias("ae"),
            F.count("*").alias("n"),
        )
        .collect()
    )
    return {int(r["t_idx"]): (float(r["se"]), float(r["ae"]), int(r["n"])) for r in rows}


def combine_sums(parts: List[Dict[int, tuple]]) -> Dict[str, float]:
    """Turn accumulated `regression_sums` into the same keys as evaluate_regression."""
    totals: Dict[int, List[float]] = {}
    for part in parts:
        for t_idx, (se, ae, n) in part.items():
            acc = totals.setdefault(t_idx, [0.0, 0.0, 0])
            acc[0] += se
            acc[1] += ae
            acc[2] += n

    if not totals:
        return {"rmse_avg": float("nan"), "mae_avg": float("nan"), "n_objects": 0.0}

    ordered = [totals[k] for k in sorted(totals)]
    rmses = [(se / n) ** 0.5 for se, _, n in ordered]
    maes = [ae / n for _, ae, n in ordered]
    total_n = sum(n for _, _, n in ordered)

    out: Dict[str, float] = {
        "rmse_avg": sum(rmses) / len(rmses),
        "mae_avg": sum(maes) / len(maes),
        "rmse_pooled": (sum(se for se, _, _ in ordered) / total_n) ** 0.5,
        "mae_pooled": sum(ae for _, ae, _ in ordered) / total_n,
        "n_targets": float(len(ordered)),
        "n_objects": float(total_n / len(ordered)),
    }
    if len(ordered) > 1:
        for i, (rmse, mae) in enumerate(zip(rmses, maes)):
            out[f"rmse_target_{i}"] = rmse
            out[f"mae_target_{i}"] = mae
    return out
