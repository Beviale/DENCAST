"""Anomaly scores derived from a fitted DENCAST model.

The regression task is left exactly as it is. DENCAST already routes an object
to a cluster and predicts its targets as the similarity-weighted average of
that cluster's members; the only thing added here is an aggregation over the
same clusters.

The idea, in one line: in the paper a good prediction is the goal, here a bad
prediction is the signal. An object whose target departs from the consensus of
the objects it most resembles is anomalous -- a sensor reporting 1.2 kWh at
noon while every comparable plant reports 5.8 is perfectly ordinary in feature
space and perfectly attached to the graph, so no topological score would ever
notice it.

    score(x) = | y(x) - prediction(x) |  /  sigma(cluster of x)

Dividing by the cluster's own spread is what makes the score comparable across
clusters: a deviation of 0.05 means something different in a tight cluster than
in a loose one, and the normalisation also compensates for the fact that a
single global threshold -- DBSCAN's weak point -- cannot fit regions of
different density.

**Why shrinkage.** The catch is that sigma has to be estimated from the cluster
members, and DENCAST's clusters are small: on PV Italy, with the configuration
that reproduces the paper, the median cluster holds 5 objects, half hold fewer
than 5, and 63 of 550 are singletons where the sample variance does not exist
at all. A z-score divided by a sigma estimated from three points is dominated
by the noise of that estimate, not by the signal. So the per-cluster variance
is shrunk towards the global one:

    sigma^2_shrunk = (n * sigma^2_cluster + k * sigma^2_global) / (n + k)

With a large cluster the estimate is essentially its own; with a tiny one it
falls back to the global spread. `k` is a pseudo-count: it is the cluster size
at which the two contribute equally.
"""

from __future__ import annotations

from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from dencast.calibration import ExpectedResidual, similarity_to_distance
from dencast.clustering import NOISE_CLUSTER_ID
from dencast.loo import cluster_moments, loo_stats

DEFAULT_SHRINKAGE = 2.0
"""Pseudo-count for the shrinkage prior: the cluster size at which the local
and the global estimate weigh the same.

Choosing it is a trade-off in both directions, and the cluster sizes DENCAST
actually produces decide where the balance sits. With the configuration that
reproduces the paper on PV Italy the median cluster holds 5 objects, so the
weight the prior carries at that size is:

    k = 1  ->  17%      k = 3  ->  38%
    k = 2  ->  29%      k = 5  ->  50%      k = 10  ->  67%

Too small and a three-member cluster that happens to agree yields a sigma near
zero and an arbitrarily large z for any residual. Too large and the prior
dominates the typical cluster: at k=10 two thirds of the variance of a median
cluster comes from the global spread, so the score stops measuring "unlike its
own neighbours" and drifts towards "unlike the dataset" -- which is exactly the
local normalisation the score exists to provide.

k=2 keeps the median cluster's own estimate in charge (71%) while still
damping the singletons. It is a default, not a constant: a run whose clusters
come out larger can afford more shrinkage, and `target_stats` takes the value
as an argument for that reason.
"""


MIN_EXPECTED_RESIDUAL = 1e-4
"""Floor on the calibration curve g.

The empirical expected residual reaches exactly 0 in the top similarity bin --
the nearest neighbour of those objects carries the same target to the digit --
and dividing by it would make every such object infinitely anomalous. The floor
is in the units of the normalised targets, so it is about a ten-thousandth of
the full range.
"""


def target_stats(
    model: DataFrame, shrinkage: float = DEFAULT_SHRINKAGE
) -> DataFrame:
    """Per-cluster, per-target mean and shrunk standard deviation.

    Args:
        model: (id, features, targets, cluster_id) as produced by `fit`.
        shrinkage: the pseudo-count `k` above. 0 disables shrinkage and gives
            the raw per-cluster estimate.

    Returns:
        (cluster_id, t_idx, mu, sigma, n) with one row per cluster and target.
        Noise objects are excluded: they belong to no cluster, so they have no
        consensus to be compared against.
    """
    clustered = model.filter(F.col("cluster_id") != F.lit(NOISE_CLUSTER_ID))

    exploded = clustered.select(
        "cluster_id", F.posexplode("targets").alias("t_idx", "t_val")
    ).cache()

    # var_samp is null for a single observation, which is exactly the case
    # shrinkage exists to handle; treating it as zero variance lets the prior
    # take over rather than propagating a null through the score.
    per_cluster = exploded.groupBy("cluster_id", "t_idx").agg(
        F.avg("t_val").alias("mu"),
        F.coalesce(F.var_samp("t_val"), F.lit(0.0)).alias("var_cluster"),
        F.count("*").alias("n"),
    )

    # The global spread of each target, over every clustered object.
    per_target = exploded.groupBy("t_idx").agg(
        F.coalesce(F.var_samp("t_val"), F.lit(0.0)).alias("var_global")
    )

    stats = (
        per_cluster.join(per_target, on="t_idx", how="left")
        .withColumn(
            "var_shrunk",
            (F.col("n") * F.col("var_cluster") + F.lit(shrinkage) * F.col("var_global"))
            / (F.col("n") + F.lit(shrinkage)),
        )
        .withColumn("sigma", F.sqrt(F.col("var_shrunk")))
        .select("cluster_id", "t_idx", "mu", "sigma", "n")
    )

    exploded.unpersist()
    return stats


def variance_scores(
    scored: DataFrame,
    stats: DataFrame,
    min_sigma: float = 1e-9,
) -> DataFrame:
    """Score every object by how far its target sits from the prediction.

    Args:
        scored: (id, cluster_id, prediction, actual). `prediction` is DENCAST's
            similarity-weighted average, not the plain cluster mean: it already
            accounts for how much the object resembles each member, so the
            residual is the sharper of the two comparisons.
        stats: output of `target_stats`.
        min_sigma: floor on sigma, to avoid dividing by zero when a target is
            constant across the whole training set.

    Returns:
        (id, cluster_id, z_max, z_mean, residual_max, sigma_mean, cluster_n).

        Two aggregations over the k targets are returned because they catch
        different failures: `z_max` reacts to one badly wrong component -- a
        single hour of a production curve -- while `z_mean` reacts to a curve
        that is off everywhere by a little. In the single-target case they
        coincide.
    """
    per_component = scored.select(
        "id",
        "cluster_id",
        F.posexplode(
            F.zip_with(
                F.col("actual"), F.col("prediction"), lambda a, p: F.abs(a - p)
            )
        ).alias("t_idx", "residual"),
    )

    joined = per_component.join(stats, on=["cluster_id", "t_idx"], how="left")

    with_z = joined.withColumn(
        "z",
        F.col("residual")
        / F.greatest(F.coalesce(F.col("sigma"), F.lit(min_sigma)), F.lit(min_sigma)),
    )

    return with_z.groupBy("id", "cluster_id").agg(
        F.max("z").alias("z_max"),
        F.avg("z").alias("z_mean"),
        F.max("residual").alias("residual_max"),
        F.avg("sigma").alias("sigma_mean"),
        F.first("n").alias("cluster_n"),
    )


def score_predictions(
    model: DataFrame,
    scored: DataFrame,
    shrinkage: float = DEFAULT_SHRINKAGE,
) -> DataFrame:
    """Convenience wrapper: compute the statistics and apply them in one call."""
    stats = target_stats(model, shrinkage=shrinkage).cache()

    n_clusters = stats.select("cluster_id").distinct().count()
    small = stats.filter(F.col("n") < 10).select("cluster_id").distinct().count()
    logger.info(
        "Target statistics over {} clusters; {} ({:.0%}) hold fewer than 10 "
        "members, which is what the shrinkage (k={}) is there to absorb",
        n_clusters,
        small,
        small / n_clusters if n_clusters else 0.0,
        shrinkage,
    )

    result = variance_scores(scored, stats)
    stats.unpersist()
    return result


def summarize(scores: DataFrame, top_n: int = 10) -> None:
    """Log the highest-scoring objects and the shape of the distribution."""
    quantiles = scores.approxQuantile("z_max", [0.5, 0.9, 0.99], 0.01)
    logger.info(
        "z_max distribution: median {:.2f}, p90 {:.2f}, p99 {:.2f}",
        *quantiles,
    )
    logger.info("Highest {} z_max:", top_n)
    for row in scores.orderBy(F.desc("z_max")).limit(top_n).collect():
        logger.info(
            "  id={} cluster={} (n={}) z_max={:.2f} residual={:.4f} sigma={:.4f}",
            row["id"],
            row["cluster_id"],
            row["cluster_n"],
            row["z_max"],
            row["residual_max"],
            row["sigma_mean"],
        )


def attach_cluster_sizes(scores: DataFrame, model: DataFrame) -> DataFrame:
    """Add the size of each object's cluster.

    Small clusters are themselves an anomaly signal -- twenty coordinated
    events form a dense micro-cluster whose members all pass any degree test --
    so the size is worth carrying alongside the variance score rather than
    folding into it.
    """
    sizes = (
        model.filter(F.col("cluster_id") != F.lit(NOISE_CLUSTER_ID))
        .groupBy("cluster_id")
        .agg(F.count("*").alias("cluster_size"))
    )
    return scores.join(sizes, on="cluster_id", how="left")

# ---------------------------------------------------------------------------
# The full score: residual, normalised twice
# ---------------------------------------------------------------------------


def full_scores(
    model: DataFrame,
    scored: DataFrame,
    expected: ExpectedResidual,
    shrinkage: float = DEFAULT_SHRINKAGE,
    leave_one_out: bool = True,
    min_sigma: float = 1e-9,
) -> DataFrame:
    """Score objects by how far their error exceeds what is normal for them.

                        | y - prediction |
        score  =  --------------------------------
                    sigma_loo(cluster)  *  g(d)

    Three quantities, each answering a different objection to the one before:

      * the **residual** alone says nothing, because some targets are simply
        harder to predict than others;
      * dividing by **sigma of the cluster** fixes that -- a deviation of 0.05
        is unremarkable among objects that scatter by 1.0 and glaring among
        objects that agree to within 0.01 -- but an object sitting far from
        everything would still light up merely for being far;
      * dividing by **g(d)** fixes that in turn: it is the error normally seen
        at that level of assignment confidence, so what is left is the part of
        the error that the model had no excuse for.

    The sigma is computed leaving the object itself out, since with the model
    trained on everything each object helped form the statistics it is judged
    against.

    Args:
        model: (id, features, targets, cluster_id) from `fit`.
        scored: (id, cluster_id, prediction, actual, best_sim) -- the output of
            `predict` joined with the withheld targets.
        expected: the calibration curve, from `fit_expected_residual`.
        shrinkage: pseudo-count for the variance prior.
        leave_one_out: False when the scored objects were not in the training set.
        min_sigma: floor on sigma, for a target that is constant everywhere.

    Returns:
        (id, cluster_id, score_max, score_mean, z_max, residual_max,
         sigma_mean, expected_residual, best_sim, n_eff)

        `z_*` is the score without the g term, kept alongside on purpose: with
        the similarities this saturated, g may turn out to add little, and
        having both means that can be measured rather than assumed.
    """
    moments = cluster_moments(model)
    stats = loo_stats(
        scored.select("id", "cluster_id", "actual"),
        moments,
        shrinkage=shrinkage,
        leave_one_out=leave_one_out,
    )

    residuals = scored.select(
        "id",
        "best_sim",
        F.posexplode(
            F.zip_with(F.col("actual"), F.col("prediction"), lambda a, p: F.abs(a - p))
        ).alias("t_idx", "residual"),
    )

    joined = residuals.join(stats, on=["id", "t_idx"], how="inner").withColumn(
        "g", expected.apply(similarity_to_distance(F.col("best_sim")))
    )

    with_scores = joined.withColumn(
        "sigma_eff", F.greatest(F.coalesce(F.col("sigma"), F.lit(min_sigma)), F.lit(min_sigma))
    ).withColumn(
        "z", F.col("residual") / F.col("sigma_eff")
    ).withColumn(
        "score", F.col("residual") / (F.col("sigma_eff") * F.col("g"))
    )

    return with_scores.groupBy("id", "cluster_id").agg(
        F.max("score").alias("score_max"),
        F.avg("score").alias("score_mean"),
        F.max("z").alias("z_max"),
        F.max("residual").alias("residual_max"),
        F.avg("sigma_eff").alias("sigma_mean"),
        F.first("g").alias("expected_residual"),
        F.first("best_sim").alias("best_sim"),
        F.min("n_eff").alias("n_eff"),
    )
