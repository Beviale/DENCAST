"""Step 4 of DENCAST: using the clusters to predict the target attributes.

This is what separates DENCAST from a plain clustering algorithm. Clusters are
not only descriptive: they become the model. For an unlabeled object u the
paper (Algorithm 4, Eq. 1 and 2) does two things:

    1. find the most similar labeled object l and inherit its cluster,
       c(u) = c(argmax_l cosine(u, l[1:m])); the nearest neighbour is used only
       as a pointer into the cluster structure, not as the predictor itself;

    2. predict u's targets as the similarity-weighted average of the targets of
       *all* the labeled objects falling in c(u).

Step 2 is what makes the prediction robust: a single nearest neighbour can be
noisy or anomalous, whereas the cluster provides a whole coherent population to
average over. In the multi-target setting the averaged vectors are real,
internally consistent observations, so their average stays plausible -- which
is why the paper reports its largest gains there.

Note that the similarity in Eq. 1 and 2 uses only the m descriptive
attributes, since u's targets are exactly what is unknown. The graph, however,
was built on the full (m + k) vector. The method therefore builds clusters in
one space and looks them up in another; the paper does not discuss this
asymmetry.

The weighted average is computed in a fully distributed way by keeping two
running sums instead of gathering the cluster members in one place:

    numerator   = sum of sim * target   (a vector, summed element-wise)
    denominator = sum of sim            (a scalar)

Both are associative and commutative, so Spark can combine them locally and
merge the partial results in any order; only the final division needs all the
partial sums, and it happens once.

Implementation note: this module follows the paper, not the Scala code. The
Scala implementation actually compares the test row against the *centroids* of
the clusters instead of against all labeled objects, which makes its prediction
step equivalent to K-means. Following the paper costs an all-pairs comparison,
which is the most expensive stage of the pipeline.
"""

from __future__ import annotations

from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from dencast.clustering import NOISE_CLUSTER_ID
from dencast.spark_session import cosine_similarity


def _global_target_mean(model: DataFrame, k: int) -> list:
    """Average of every target over the whole training set.

    Used as a fallback for the rare unlabeled object whose nearest labeled
    neighbour belongs to no cluster. It is the same thing as the AVG baseline
    of the paper.
    """
    exploded = model.select(F.posexplode("targets").alias("t_idx", "t_val"))
    rows = (
        exploded.groupBy("t_idx")
        .agg(F.avg("t_val").alias("mean"))
        .orderBy("t_idx")
        .collect()
    )
    return [row["mean"] for row in rows]


def predict(
    unlabeled: DataFrame,
    model: DataFrame,
    k: int,
    include_noise_as_anchor: bool = False,
) -> DataFrame:
    """Predict the k targets of every unlabeled object.

    Args:
        unlabeled: (id, features) where features holds the m descriptive
            attributes.
        model: (id, features, targets, cluster_id), the labeled objects with
            the cluster they were assigned to.
        k: number of target attributes.
        include_noise_as_anchor: whether an object labelled as noise may be
            chosen as the nearest neighbour. Default False: noise objects
            belong to no cluster, so there would be no population to average
            over.

    Returns:
        (id, prediction, cluster_id, best_sim) with prediction an
        array<double> of length k, cluster_id the cluster the object was routed
        to, and best_sim the similarity of the nearest labeled object.
    """
    candidates = model
    if not include_noise_as_anchor:
        candidates = model.filter(F.col("cluster_id") != F.lit(NOISE_CLUSTER_ID))

    n_unlabeled = unlabeled.count()
    n_candidates = candidates.count()
    if n_candidates == 0:
        raise ValueError(
            "No labeled object belongs to a cluster: nothing to predict from. "
            "Lower min_pts or min_sim so that the graph has core objects."
        )
    logger.info(
        "Prediction: %d unlabeled x %d labeled = %d pairs to score",
        n_unlabeled,
        n_candidates,
        n_unlabeled * n_candidates,
    )

    u = unlabeled.select(
        F.col("id").alias("u_id"), F.col("features").alias("u_features")
    )
    labeled = candidates.select(
        F.col("features").alias("l_features"),
        F.col("targets").alias("l_targets"),
        F.col("cluster_id"),
    )

    # Eq. 1 and 2 both need the similarity between u and every labeled object,
    # computed on the descriptive attributes only.
    sims = u.crossJoin(labeled).withColumn(
        "sim", cosine_similarity(F.col("u_features"), F.col("l_features"))
    )

    # Eq. 1: the cluster of the most similar labeled object. Taking the max of
    # a struct compares the fields in order, so ties on the similarity are
    # broken deterministically by the cluster id.
    best = (
        sims.groupBy("u_id")
        .agg(F.max(F.struct("sim", "cluster_id")).alias("best"))
        .select(
            "u_id",
            F.col("best.cluster_id").alias("cluster_id"),
            F.col("best.sim").alias("best_sim"),
        )
    )

    # Keep only the labeled objects that fall in the chosen cluster.
    members = sims.join(best, on=["u_id", "cluster_id"], how="inner")

    # Eq. 2, element-wise and distributed: explode the target vectors into
    # (object, target index, value), accumulate the two sums per index, divide
    # once at the end, then reassemble the vector in index order.
    exploded = members.select(
        "u_id", "sim", F.posexplode("l_targets").alias("t_idx", "t_val")
    )

    per_component = (
        exploded.groupBy("u_id", "t_idx")
        .agg(
            F.sum(F.col("sim") * F.col("t_val")).alias("numerator"),
            F.sum("sim").alias("denominator"),
        )
        .withColumn(
            "value",
            F.when(F.col("denominator") != 0, F.col("numerator") / F.col("denominator"))
            .otherwise(F.lit(0.0)),
        )
    )

    assembled = (
        per_component.groupBy("u_id")
        .agg(F.sort_array(F.collect_list(F.struct("t_idx", "value"))).alias("pairs"))
        .select(
            "u_id",
            F.transform(F.col("pairs"), lambda p: p["value"]).alias("prediction"),
        )
        # The cluster the object was routed to is carried through: anything
        # built on top of the prediction -- an uncertainty estimate, a
        # variance-based anomaly score -- needs to know which population the
        # prediction came from, and recomputing Eq. 1 to get it back would mean
        # redoing the all-pairs comparison.
        .join(best.select("u_id", "cluster_id", "best_sim"), on="u_id", how="left")
    )

    # An unlabeled object can be left without a prediction only if it had no
    # comparable labeled object at all; fall back to the training mean.
    fallback = _global_target_mean(model, k)
    fallback_col = F.array(*[F.lit(float(v)) for v in fallback])

    result = (
        unlabeled.select(F.col("id"))
        .join(assembled, unlabeled["id"] == assembled["u_id"], how="left")
        .select(
            F.col("id"),
            F.coalesce(F.col("prediction"), fallback_col).alias("prediction"),
            F.coalesce(F.col("cluster_id"), F.lit(NOISE_CLUSTER_ID)).alias("cluster_id"),
            # The similarity of the nearest labeled object, i.e. how firmly the
            # object was routed to that cluster. Carried through because it
            # calibrates how much error to expect: a residual that would be
            # unremarkable next to a distant neighbour is surprising next to a
            # near-identical one.
            F.coalesce(F.col("best_sim"), F.lit(0.0)).alias("best_sim"),
        )
    )

    return result


def attach_ground_truth(
    predictions: DataFrame, unlabeled_with_targets: DataFrame
) -> DataFrame:
    """Join the predictions with the withheld true targets, for evaluation.

    "Unlabeled" means hidden from the algorithm, not unknown to us: the targets
    were held out on purpose so that the predictions can be scored.
    """
    truth = unlabeled_with_targets.select(
        F.col("id"), F.col("targets").alias("actual")
    )
    return predictions.join(truth, on="id", how="inner")


# ---------------------------------------------------------------------------
# DVC stage: rolling evaluation over the test split
# ---------------------------------------------------------------------------
