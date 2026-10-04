"""Steps 2-3 of DENCAST: core objects and density-based clustering.

Core objects are identified exactly as in DBSCAN, except that the neighbourhood
is read straight off the graph instead of being recomputed: an object is a core
object when its degree is at least minPts.

The clustering itself is the contribution of the paper. DBSCAN grows one
cluster at a time from an arbitrary core object; DENCAST instead expands all
core objects simultaneously by propagating cluster IDs along the edges, which
removes the need for the final merging phase that other distributed
density-based methods run on a single machine.

Each iteration is one map and one reduce (Algorithm 2 of the paper):

    map:     for every edge <src, dst>, if src is a core object and its cluster
             ID is greater than dst's, send src's ID to dst;
    reduce:  every node keeps the maximum of the IDs it received.

Two consequences of that design are worth keeping in mind:

  * the `srcID > dstID` guard makes IDs monotonically non-decreasing, and since
    they are bounded by the largest initial ID the process must converge;
  * `max` is associative and commutative, so Spark can combine messages locally
    before shuffling them, and the result does not depend on the order in which
    partitions are processed.

Because the density-connection relation is symmetric and transitive, the fixed
point of this process is the set of connected components of the subgraph
reachable from core objects. Propagating the smallest IDs instead of the
largest would give the same partition with different names.

GraphX is not exposed in PySpark, so the message passing is implemented here
with DataFrame joins and aggregations rather than `aggregateMessages`.
"""

import logging
from typing import Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from dencast.utils import Params

logger = logging.getLogger(__name__)

NOISE_CLUSTER_ID = 0
"""Cluster ID reserved for noise: objects that are neither core objects nor
neighbours of one. Like DBSCAN, DENCAST leaves them out of every cluster."""


def identify_core_objects(
    vertices: DataFrame, edges: DataFrame, min_pts: int
) -> DataFrame:
    """Return the vertices with a boolean `is_core` column.

    Since `edges` holds both directions, counting the outgoing edges of a node
    already gives |N(p)|.
    """
    degrees = edges.groupBy("src").agg(F.count("*").alias("degree"))

    result = (
        vertices.join(degrees, vertices["id"] == degrees["src"], how="left")
        .select(
            F.col("id"),
            F.coalesce(F.col("degree"), F.lit(0)).alias("degree"),
        )
        .withColumn("is_core", F.col("degree") >= F.lit(min_pts))
    )
    return result


def _initialize_cluster_ids(vertices_with_core: DataFrame) -> DataFrame:
    """Give every core object its own cluster ID, and 0 to the others.

    The paper assigns 1, 2, 3, ...; the Scala code reuses the vertex id. Both
    work, because only the relative order of the IDs matters. Here we use
    `id + 1` so that the value is unique, deterministic, free to compute, and
    never collides with the reserved noise ID of 0.
    """
    return vertices_with_core.withColumn(
        "cluster_id",
        F.when(F.col("is_core"), F.col("id") + F.lit(1)).otherwise(
            F.lit(NOISE_CLUSTER_ID)
        ),
    ).select("id", "is_core", "cluster_id")


def find_clusters(
    vertices: DataFrame, edges: DataFrame, params: Params
) -> Tuple[DataFrame, int]:
    """Run the distributed density-based clustering (Algorithm 2).

    Returns:
        (clusters, n_iterations) where `clusters` is (id, is_core, cluster_id).
        A cluster_id of 0 marks a noise object.
    """
    clu_cfg = params.clustering

    with_core = identify_core_objects(vertices, edges, clu_cfg.min_pts).cache()
    n_core = with_core.filter(F.col("is_core")).count()
    logger.info("Core objects (degree >= %d): %d", clu_cfg.min_pts, n_core)

    if n_core == 0:
        logger.warning(
            "No core object found: every object will be labelled as noise. "
            "Lower min_pts or min_sim to get a denser graph."
        )

    n_edges = edges.count()
    threshold = clu_cfg.label_change_rate * n_edges
    logger.info(
        "Stopping threshold: %.1f propagations (%.0f%% of %d edges)",
        threshold,
        100 * clu_cfg.label_change_rate,
        n_edges,
    )

    state = _initialize_cluster_ids(with_core).localCheckpoint(eager=True).cache()
    state.count()  # materialize before entering the loop
    with_core.unpersist()

    # Only the endpoints matter for the message passing; the similarity carried
    # by the edges is not used by the propagation, which treats every edge the
    # same way.
    plain_edges = edges.select("src", "dst").cache()

    iteration = 0
    for iteration in range(1, clu_cfg.max_iterations + 1):
        src_state = state.select(
            F.col("id").alias("src"),
            F.col("is_core").alias("src_is_core"),
            F.col("cluster_id").alias("src_cid"),
        )
        dst_state = state.select(
            F.col("id").alias("dst"),
            F.col("cluster_id").alias("dst_cid"),
        )

        # MAP: a core object offers its cluster ID to every neighbour that is
        # currently holding a smaller one.
        messages = (
            plain_edges.join(src_state, on="src")
            .join(dst_state, on="dst")
            .filter(F.col("src_is_core") & (F.col("src_cid") > F.col("dst_cid")))
            .select(F.col("dst").alias("id"), F.col("src_cid").alias("cid"))
        )

        # REDUCE: every node keeps the largest ID it was offered. The message
        # count is carried through the same aggregation, so the stopping
        # criterion costs no extra pass over the messages -- and because both
        # max and sum are associative and commutative, Spark combines them
        # locally before shuffling.
        best = (
            messages.groupBy("id")
            .agg(F.max("cid").alias("new_cid"), F.count("*").alias("n_msgs"))
            .cache()
        )

        propagations = best.agg(F.sum("n_msgs")).collect()[0][0] or 0
        logger.info("Iteration %d: %d propagations", iteration, propagations)

        if propagations < threshold:
            best.unpersist()
            break

        new_state = (
            state.join(best, on="id", how="left")
            .withColumn(
                "cluster_id",
                F.when(
                    F.col("new_cid").isNotNull() & (F.col("new_cid") > F.col("cluster_id")),
                    F.col("new_cid"),
                ).otherwise(F.col("cluster_id")),
            )
            .select("id", "is_core", "cluster_id")
        )

        # The loop rebuilds the plan on top of the previous one, so without
        # truncating the lineage Spark ends up re-deriving the whole history at
        # every action, and eventually overflows the stack when analysing it.
        new_state = new_state.localCheckpoint(eager=True)

        state.unpersist()
        best.unpersist()
        state = new_state.cache()

    plain_edges.unpersist()

    if iteration >= clu_cfg.max_iterations:
        logger.warning(
            "Reached max_iterations=%d without going below the threshold",
            clu_cfg.max_iterations,
        )

    return state, iteration


def cluster_summary(clusters: DataFrame) -> DataFrame:
    """Size of every extracted cluster, noise excluded, largest first."""
    return (
        clusters.filter(F.col("cluster_id") != F.lit(NOISE_CLUSTER_ID))
        .groupBy("cluster_id")
        .agg(F.count("*").alias("size"))
        .orderBy(F.desc("size"))
    )
