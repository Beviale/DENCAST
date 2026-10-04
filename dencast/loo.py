"""Leave-one-out cluster statistics.

Training on the whole dataset, anomalies included, makes the scoring
transductive: every object took part in forming the cluster it is then
compared against, so it contributed to the very mean and variance used to
judge it. A strong anomaly inflates the sigma it is divided by, and so lowers
its own score -- it masks itself. With DENCAST's clusters holding five to
eight members the leverage of a single point is large enough that this is not
a detail but the difference between catching an anomaly and not.

Removing the point is closed-form on the sufficient statistics, so nothing is
recomputed per object:

    n'    = n - 1
    mean' = (s1 - y) / n'
    var'  = (s2 - y^2 - (s1 - y)^2 / n') / (n' - 1)

Clusters of one or two members leave too little behind for a variance; those
fall through to the shrinkage prior, which is exactly what it is for.

This is only needed when the scored object is itself part of the training set.
For a genuinely new object the plain statistics are already leave-one-out.
"""

from __future__ import annotations

from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from dencast.clustering import NOISE_CLUSTER_ID


def cluster_moments(model: DataFrame) -> DataFrame:
    """Per-cluster, per-target sufficient statistics: count, sum, sum of squares.

    Every later quantity -- mean, variance, and their leave-one-out versions --
    is arithmetic on these three, so the data is scanned once and the rest is
    joins.
    """
    clustered = model.filter(F.col("cluster_id") != F.lit(NOISE_CLUSTER_ID))
    exploded = clustered.select(
        "cluster_id", F.posexplode("targets").alias("t_idx", "t_val")
    )
    return exploded.groupBy("cluster_id", "t_idx").agg(
        F.count("*").alias("n"),
        F.sum("t_val").alias("s1"),
        F.sum(F.col("t_val") * F.col("t_val")).alias("s2"),
    )


def global_variance(moments: DataFrame) -> DataFrame:
    """Per-target variance over every clustered object: the shrinkage prior."""
    return moments.groupBy("t_idx").agg(
        (
            (F.sum("s2") - F.sum("s1") * F.sum("s1") / F.sum("n"))
            / F.greatest(F.sum("n") - F.lit(1), F.lit(1))
        ).alias("var_global")
    )


def loo_stats(
    scored: DataFrame,
    moments: DataFrame,
    shrinkage: float,
    var_global: Optional[DataFrame] = None,
    leave_one_out: bool = True,
) -> DataFrame:
    """Per-object, per-target cluster statistics with the object removed.

    Args:
        scored: (id, cluster_id, actual, ...). `actual` is the true target
            vector, which is what gets subtracted out.
        moments: output of `cluster_moments`.
        shrinkage: pseudo-count k for shrinking towards the global variance.
        var_global: optional precomputed prior; derived from `moments` if absent.
        leave_one_out: set False when scoring objects that were not part of the
            training set, where the plain statistics are already correct.

    Returns:
        (id, cluster_id, t_idx, y, mu, sigma, n_eff).
    """
    if var_global is None:
        var_global = global_variance(moments)

    per_component = scored.select(
        "id", "cluster_id", F.posexplode("actual").alias("t_idx", "y")
    )

    joined = per_component.join(moments, on=["cluster_id", "t_idx"], how="left").join(
        var_global, on="t_idx", how="left"
    )

    if leave_one_out:
        n_eff = F.col("n") - F.lit(1)
        s1_eff = F.col("s1") - F.col("y")
        s2_eff = F.col("s2") - F.col("y") * F.col("y")
    else:
        n_eff = F.col("n")
        s1_eff = F.col("s1")
        s2_eff = F.col("s2")

    result = (
        joined.withColumn("n_eff", n_eff)
        .withColumn(
            "mu",
            F.when(F.col("n_eff") > 0, s1_eff / F.col("n_eff")),
        )
        .withColumn(
            "var_own",
            F.when(
                F.col("n_eff") > 1,
                # Clamped at zero: catastrophic cancellation on nearly
                # identical values can push the expression a hair negative.
                F.greatest(
                    (s2_eff - s1_eff * s1_eff / F.col("n_eff"))
                    / (F.col("n_eff") - F.lit(1)),
                    F.lit(0.0),
                ),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "var_shrunk",
            (
                F.coalesce(F.col("n_eff"), F.lit(0)) * F.col("var_own")
                + F.lit(shrinkage) * F.coalesce(F.col("var_global"), F.lit(0.0))
            )
            / (F.coalesce(F.col("n_eff"), F.lit(0)) + F.lit(shrinkage)),
        )
    )

    return result.select(
        "id",
        "cluster_id",
        "t_idx",
        "y",
        "mu",
        F.sqrt(F.col("var_shrunk")).alias("sigma"),
        F.coalesce(F.col("n_eff"), F.lit(0)).alias("n_eff"),
    )
